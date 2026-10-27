"""JSON contracts and conservative candidate screening (not a security proof)."""
import ast
import hashlib
import json
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_task(task):
    required = {"task_id", "family", "split", "prompt", "reference_code", "input_specs", "test_cases", "atol", "rtol"}
    if not required <= task.keys():
        raise ValueError(f"Missing task keys: {required - task.keys()}")
    if task["split"] not in {"train", "validation", "test"}:
        raise ValueError("Invalid split")
    if not task["input_specs"] or len(task["test_cases"]) < 3:
        raise ValueError("Require inputs and at least three trusted test cases")
    if any(spec["dtype"] != "float32" for spec in task["input_specs"]):
        raise ValueError("Initial contract supports float32 only")
    if task["atol"] < 0 or task["rtol"] < 0:
        raise ValueError("Negative tolerance")
    return task


class _HostChecker:
    """Conservative taint checking: host code allocates and handles metadata only.

    Deliberately supports a small wrapper language. Unsupported Python is rejected
    rather than guessed safe. This checks compute placement, not hostile-code safety.
    """
    scalar_calls = {"int", "float", "bool", "len", "min", "max", "abs", "range"}
    metadata = {"shape", "ndim", "device", "dtype", "is_cuda"}
    metadata_calls = {"numel", "stride", "is_contiguous", "size", "dim"}

    def __init__(self, tree):
        self.env = {name: "scalar_call" for name in self.scalar_calls}
        self.jit_names = set()
        self.saw_launch = self.saw_return = False
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    self.env[alias.asname or alias.name.split(".")[0]] = "module:" + alias.name
            elif isinstance(statement, ast.FunctionDef) and statement.name != "kernel":
                decorators = statement.decorator_list
                if len(decorators) != 1 or not self.is_jit(decorators[0]):
                    raise ValueError("Host helper functions are unsupported; use a metadata lambda for launch grids")
                self.env[statement.name] = "jit"
                self.jit_names.add(statement.name)
        wrappers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "kernel"]
        if len(wrappers) != 1:
            raise ValueError("Require exactly one def kernel wrapper")
        for statement in tree.body:
            if isinstance(statement, (ast.Import, ast.FunctionDef)):
                continue
            self.statement(statement, self.env, module_scope=True)
        wrapper = wrappers[0]
        if wrapper.decorator_list or wrapper.args.vararg or wrapper.args.kwarg or wrapper.args.kwonlyargs or wrapper.args.defaults:
            raise ValueError("kernel must have plain positional tensor inputs without decorators or defaults")
        env = dict(self.env)
        for argument in wrapper.args.posonlyargs + wrapper.args.args:
            if argument.arg in self.env:
                raise ValueError("Wrapper input shadows a reserved name")
            env[argument.arg] = "tensor"
        for statement in wrapper.body:
            self.statement(statement, env)
        if not self.saw_launch or not self.saw_return:
            raise ValueError("Wrapper must launch a declared Triton kernel and return a tensor")

    @staticmethod
    def is_jit(node):
        if isinstance(node, ast.Call):
            node = node.func
        return isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "triton" and node.attr == "jit"

    @staticmethod
    def scalar(kind):
        if kind != "scalar":
            raise ValueError("Tensor arithmetic, indexing, iteration, and computed host values are forbidden outside Triton JIT")

    def expression(self, node, env):
        if isinstance(node, ast.Constant):
            return "scalar"
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise ValueError(f"Unsupported or unbound host name: {node.id}")
            return env[node.id]
        if isinstance(node, ast.Attribute):
            base = self.expression(node.value, env)
            if base == "tensor":
                if node.attr in self.metadata:
                    return "scalar"
                if node.attr in self.metadata_calls:
                    return "metadata_call"
                raise ValueError(f"Tensor method/property {node.attr} is forbidden outside Triton JIT")
            if base == "module:torch":
                if node.attr in {"empty", "zeros"}:
                    return "allocation"
                if node.attr in {"empty_like", "zeros_like"}:
                    return "allocation_like"
                if node.attr in {"float32", "Tensor"}:
                    return "scalar"
            if base == "module:triton" and node.attr in {"cdiv", "next_power_of_2"}:
                return "scalar_call"
            raise ValueError(f"Unsupported host attribute: {node.attr}")
        if isinstance(node, ast.Subscript):
            base = self.expression(node.value, env)
            self.scalar(self.expression(node.slice, env))
            if base == "jit":
                return "launch"
            self.scalar(base)
            return "scalar"
        if isinstance(node, ast.Slice):
            for value in (node.lower, node.upper, node.step):
                if value is not None:
                    self.scalar(self.expression(value, env))
            return "scalar"
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            for value in node.elts:
                self.scalar(self.expression(value, env))
            return "scalar"
        if isinstance(node, ast.Dict):
            for value in node.keys + node.values:
                if value is not None:
                    self.scalar(self.expression(value, env))
            return "scalar"
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    self.scalar(self.expression(child, env))
            return "scalar"
        if isinstance(node, ast.IfExp):
            self.scalar(self.expression(node.test, env))
            left, right = self.expression(node.body, env), self.expression(node.orelse, env)
            if left != right:
                raise ValueError("Host conditional changes tensor/scalar type")
            return left
        if isinstance(node, ast.Lambda):
            if node.args.vararg or node.args.kwarg or node.args.defaults:
                raise ValueError("Unsupported launch grid lambda")
            local = dict(env)
            for argument in node.args.args:
                local[argument.arg] = "scalar"
            self.scalar(self.expression(node.body, local))
            return "scalar"  # Triton consumes this only as launch-grid metadata.
        if isinstance(node, ast.Call):
            function = self.expression(node.func, env)
            arguments = [self.expression(value, env) for value in node.args]
            keywords = [self.expression(value.value, env) for value in node.keywords]
            if function == "launch":
                if any(kind not in {"tensor", "scalar"} for kind in arguments + keywords):
                    raise ValueError("Unsupported Triton launch argument")
                self.saw_launch = True
                return "none"
            if function == "allocation_like":
                if not arguments or arguments[0] != "tensor":
                    raise ValueError("*_like allocation requires a tensor")
                arguments = arguments[1:]
            if function in {"allocation", "allocation_like", "metadata_call", "scalar_call"}:
                for kind in arguments + keywords:
                    self.scalar(kind)
                return "tensor" if function.startswith("allocation") else "scalar"
            raise ValueError("Only allocations, metadata calls, and Triton launches are allowed in host code")
        raise ValueError(f"Unsupported host expression: {type(node).__name__}")

    def assign(self, target, kind, env):
        if isinstance(target, ast.Name):
            if target.id in self.env or target.id in self.jit_names:
                raise ValueError("Cannot overwrite imports, builtins, or Triton kernel names")
            env[target.id] = kind
        elif isinstance(target, (ast.Tuple, ast.List)):
            self.scalar(kind)
            for element in target.elts:
                self.assign(element, "scalar", env)
        else:
            raise ValueError("Host tensor/subscript/attribute assignments are forbidden")

    def statement(self, node, env, module_scope=False):
        if isinstance(node, ast.Expr):
            if module_scope and not isinstance(node.value, ast.Constant):
                raise ValueError("Executable module-scope expressions are forbidden")
            self.expression(node.value, env)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            if node.value is None:
                raise ValueError("Host assignment needs a value")
            kind = self.expression(node.value, env)
            if module_scope:
                self.scalar(kind)
            for target in node.targets if isinstance(node, ast.Assign) else [node.target]:
                self.assign(target, kind, env)
        elif not module_scope and isinstance(node, ast.AugAssign):
            self.scalar(self.expression(node.target, env))
            self.scalar(self.expression(node.value, env))
        elif not module_scope and isinstance(node, ast.Return):
            if node.value is None or self.expression(node.value, env) != "tensor":
                raise ValueError("kernel must return a tensor allocated by the wrapper")
            self.saw_return = True
        elif not module_scope and isinstance(node, ast.Assert):
            self.scalar(self.expression(node.test, env))
            if node.msg:
                self.scalar(self.expression(node.msg, env))
        elif not module_scope and isinstance(node, ast.If):
            self.scalar(self.expression(node.test, env))
            branches = [dict(env), dict(env)]
            for statements, branch in zip([node.body, node.orelse], branches):
                for statement in statements:
                    self.statement(statement, branch)
            for name in branches[0].keys() & branches[1].keys():
                if branches[0][name] != branches[1][name]:
                    raise ValueError("Host branches change tensor/scalar type")
                env[name] = branches[0][name]
        elif isinstance(node, ast.Pass):
            pass
        else:
            raise ValueError(f"Unsupported host statement: {type(node).__name__}")


def screen_candidate(code):
    if len(code.encode()) > 100_000:
        raise ValueError("Candidate exceeds 100 KB")
    tree = ast.parse(code)
    allowed_imports = {"torch", "triton", "triton.language"}
    forbidden = {"eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "globals", "locals", "vars", "__import__", "breakpoint", "input", "print"}
    torch_attrs = {"empty", "empty_like", "zeros", "zeros_like", "float32", "Tensor"}
    jit = launch = wrapper = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name not in allowed_imports or alias.asname not in (None, {"torch": "torch", "triton": "triton", "triton.language": "tl"}.get(alias.name)) for alias in node.names):
                raise ValueError("Only torch/triton imports permitted")
        if isinstance(node, ast.ImportFrom):
            raise ValueError("Use import torch / import triton / import triton.language as tl")
        if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith("__")):
            raise ValueError(f"Forbidden identifier: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in {"save", "storage", "set_default_device", "manual_seed", "cuda", "cpu", "numpy", "tolist", "item", "copy_", "fill_", "zero_", "data", "detach", "frombuffer"}:
                raise ValueError(f"Forbidden attribute: {node.attr}")
            if isinstance(node.value, ast.Name) and node.value.id == "torch" and node.attr not in torch_attrs:
                raise ValueError("torch is for allocation only; compute in Triton")
            jit |= isinstance(node.value, ast.Name) and node.value.id == "triton" and node.attr == "jit"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            launch = True
        if isinstance(node, ast.FunctionDef) and node.name == "kernel":
            wrapper = True
    if not (jit and launch and wrapper):
        raise ValueError("Require Triton JIT, kernel launch, and def kernel wrapper")
    _HostChecker(tree)
    return tree

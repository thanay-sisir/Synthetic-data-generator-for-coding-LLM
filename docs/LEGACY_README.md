# Historical README — unverified claims, superseded

This preserves the old documentation for context only. Its dataset counts, performance, correctness and novelty claims are not supported by this checkout. Use PROJECT_HANDOFF.md and the root README for current status.

# Python-to-Triton LLM: Automated Code Translation

## About Project

This project is an advanced end-to-end machine learning initiative focused on developing a specialized LLM capable of automatically translating Python function bodies into highly optimized Triton kernels. It addresses the critical need to bridge the gap between Python's development agility and Triton's unparalleled performance for GPU-accelerated computations. Built upon a foundation of synthetic data.

## Architecture and Metrics

* **Synthetic Data Pipeline:**
    * A custom pipeline generates and processes over **2,000 unique synthetic Python-Triton code pairs**.
    * Each pair undergoes rigorous **automated validation** via dedicated test scripts, ensuring data integrity and correctness before inclusion in the training set.

* **Transformer LLM Architecture:**
    * At its core is a **50M-parameter Transformer-based LLM**, custom-implemented in PyTorch.
    * The model features a multi-layered decoder-only architecture, incorporating advanced components such as **Multi-Head Attention with Rotary Position Embeddings (RoPE)** and a Feed-Forward Network.

* **Training & Optimization Metrics:**
    * The training regimen employs **mixed-precision (AMP)** for accelerated computation and reduced memory footprint.
    * **Gradient accumulation** is utilized to achieve a **4x effective batch size increase** (e.g., from 24 to 96), enabling more stable training with larger effective batch sizes without requiring excessive GPU memory.
    * A novel **Muon optimizer**, a hybrid momentum-orthogonalized approach, is implemented to enhance model convergence and stability.

---

## Technical Architecture

### Model Specifications

My project implements a custom decoder-only Transformer architecture with the following specifications:

- **Model Dimension (`d_model`)**: 384
- **Attention Heads (`n_heads`)**: 8  
- **Transformer Layers (`n_layers`)**: 6
- **Feed-Forward Dimension (`d_ff`)**: 1536 (4× expansion ratio)
- **Total Parameters**: ~50M trainable parameters
- **Sequence Length**: 512 tokens maximum
- **Vocabulary**: Leverages SmolLM-135M tokenizer (49,152 tokens)

### Advanced Components

#### 1. Rotary Position Embeddings (RoPE)
My project employs RoPE for positional encoding, offering superior length extrapolation compared to absolute positional embeddings:
```
θ_i = 10000^(-2i/d) for i ∈ [0, d/2)
```
The rotary transformation applies to query and key vectors, enabling the model to capture relative positional information efficiently.

#### 2. Multi-Head Attention with Flash Attention
Utilizes PyTorch's `scaled_dot_product_attention` with optimized CUDA kernels:
- **Computational Complexity**: O(n²d) where n is sequence length
- **Memory Efficient**: Fused attention kernels reduce memory overhead
- **Causal Masking**: Ensures autoregressive property for language modeling

#### 3. Muon Optimizer (Momentum Orthogonalized by Newton-Schulz)
A hybrid optimization strategy combining:
- **Newton-Schulz Iterations**: 5-step orthogonalization via zeroth power computation
- **Momentum**: 0.95 with Nesterov acceleration
- **Adaptive LR Scaling**: `lr × √(max(1, rows/cols))` for parameter-specific learning rates
- **Hybrid Approach**: Muon for 2D weight matrices, AdamW for embeddings and normalization layers

**Muon Parameters Distribution**:
- Muon optimizer: ~45M parameters (linear projections)
- AdamW optimizer: ~5M parameters (embeddings, LayerNorms)

---

## Data Pipeline & Validation

### Synthetic Data Generation

My project's data pipeline consists of **53 unique implementation files**, each containing:
1. **Python Reference Implementation**: Vanilla PyTorch operations
2. **Triton Kernel Implementation**: GPU-optimized parallel kernels
3. **Automated Test Harness**: Numerical validation with `torch.allclose()`

### Validation Framework

The `run_all_tests.py` script executes comprehensive validation:
- **Timeout Protection**: 60-second limit per test
- **Exit Code Validation**: Only passing tests (exit code 0) are included
- **Numerical Accuracy**: Ensures Triton kernels match PyTorch outputs within tolerance

**Validated Data Statistics**:
- Total implementation files: 53
- Training pairs extracted: 319 validated examples
- Success rate: ~100% (only passing tests included in training corpus)

---

## Training Configuration

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch Size | 24 | Memory-optimized for single GPU |
| Gradient Accumulation | 4 steps | Effective batch size: 96 |
| Learning Rate (Muon) | 0.01 | Aggressive for orthogonalized gradients |
| Learning Rate (AdamW) | 0.001 | Conservative for embeddings |
| Weight Decay | 0.1 | L2 regularization |
| Dropout | 0.1 | Applied to attention and FFN |
| Gradient Clipping | 1.0 | Prevents exploding gradients |
| Epochs | 10 | Full dataset coverage |

### Learning Rate Schedule

### Mixed Precision Training

- **Automatic Mixed Precision (AMP)**: Reduces memory by ~40%
- **GradScaler**: Dynamic loss scaling for stable fp16 training
- **Precision**: bfloat16 for Newton-Schulz iterations, fp16 elsewhere

---

## Performance Metrics

### Expected Training Performance

Based on the architecture and configuration:

- **Training Time**: ~2-3 hours on NVIDIA V100/A100 (single GPU)
- **Memory Usage**: ~8-12 GB VRAM with AMP
- **Throughput**: ~500-800 tokens/second (depending on GPU)
- **Convergence**: Perplexity < 10 achievable after 5-7 epochs

### Evaluation Metrics

The model is evaluated using:
- **Cross-Entropy Loss**: Primary optimization objective
- **Token Accuracy**: Per-token prediction accuracy
- **Perplexity**: `exp(loss)` - measures model uncertainty

## Installation & Dependencies



### Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Triton
pip install triton

# Install transformers for tokenizer
pip install transformers

# Install additional dependencies
pip install pandas numpy tqdm
```

---

## Usage Guide

### 1. Generate Training Data

Extract validated Python-Triton pairs from synthetic implementations:

```bash
python extract_training_data.py
```

**Output**: `training_data.csv` with 319 training examples.

### 2. Train the Model

Run the full training pipeline:

```bash
python llm.py
```

**Training Process**:
- Loads and tokenizes training data
- Initializes 50M-parameter Transformer model
- Trains for 10 epochs with Muon optimizer
- Saves checkpoint to `trained_model.pth`
- Displays demo generations

**Expected Output**:
```
🌱 Set all seeds to 42
🔍 Device: CUDA
📊 Model Configuration:
   Architecture: 384d, 6L, 8H, 1536ff
   Training: 10 epochs, batch size 24
📊 Total parameters: 50,331,648
...
🎉 TRAINING COMPLETED!
⏱️ Total time: 120.5 minutes
🏆 Final Results:
   Validation Loss: 2.1234
   Validation Perplexity: 8.35
```

### 3. Interactive Demo

Test the trained model with custom Python functions:

```bash
python llm.py --demo
```

**Example Session**:
```
🎮 INTERACTIVE DEMO MODE
Enter Python functions and see the generated Triton kernels!

🐍 Enter Python function: y = torch.sigmoid(x)

🔄 Generating Triton kernel...

🎯 Generated Triton kernel:
pid = tl.program_id(axis=0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(input_ptr + offsets, mask=mask)
result = tl.sigmoid(x)
tl.store(output_ptr + offsets, result, mask=mask)
```

---

## Research Contributions

### Key Innovations

1. **Domain-Specific LLM**: First specialized model for Python→Triton translation
2. **Synthetic Data Validation**: Automated testing ensures 100% correctness of training data
3. **Muon Optimization**: Novel application of orthogonalized momentum for code generation
4. **Efficient Architecture**: 50M parameters achieve strong performance with minimal compute

### Applications

- **Kernel Optimization**: Automate GPU kernel development
- **Performance Acceleration**: Bridge Python-Triton gap for ML practitioners
- **Educational Tool**: Learn Triton programming patterns from Python code
- **Research Platform**: Foundation for code translation research

---

## Model Capabilities

My project demonstrates proficiency in translating:

- **Element-wise Operations**: `add`, `multiply`, `sigmoid`, `tanh`, `exp`, `log`
- **Broadcasting**: Implicit shape expansion in binary operations
- **Reduction Operations**: `sum`, `mean`, `max`, `min` across dimensions
- **Advanced Functions**: `matmul`, `softmax`, `layer_norm`, `reshape`
- **Memory Patterns**: Coalesced loads/stores with masking for boundary handling

**Example Translations** (simplified):

| Python | Generated Triton |
|--------|------------------|
| `y = x + 5` | `tl.store(y_ptr, tl.load(x_ptr) + 5)` |
| `y = torch.sigmoid(x)` | `tl.store(y_ptr, tl.sigmoid(tl.load(x_ptr)))` |
| `y = x.sum()` | Multi-stage reduction kernel with shared memory |

---

## Limitations & Future Work

### Current Limitations

- **Context Length**: Limited to 512 tokens (restricts complex kernels)
- **Training Data**: 319 examples may underfit distribution
- **Validation**: No execution-based verification of generated kernels
- **Error Handling**: No syntax/semantic checking of outputs

### Future Enhancements

1. **Scale Training Data**: Generate 10K+ validated examples
2. **Longer Context**: Extend to 2048+ tokens using sparse attention
3. **Execution Feedback**: RL-based fine-tuning with compiler feedback
4. **Multi-Task Learning**: Support PyTorch→CUDA, JAX→XLA translations
5. **Quantization**: Deploy 4-bit/8-bit models for edge inference



**Contact**: For questions or collaboration, reach out via GitHub Issues.

**Status**: 🚀 Active Development | 🔬 Research Project

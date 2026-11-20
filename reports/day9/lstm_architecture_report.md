# PerceptGuard AI — Day 9 Task 3: LSTM Architecture Development Report

**Date:** 2026-11-20  
**Phase:** Day 9 Task 3  
**Audit Scope:** LSTM Forecasting Architecture Development, Input Validation, CPU Optimization, Training Interface Compatibility, and Empirical Verification  
**Lead Roles:** Senior Staff Machine Learning Engineer, Deep Learning Architect  
**Repository:** `https://github.com/Jeeva1117/PerceptGuard-AI.git`  
**Branch:** `main`  
**Execution Environment:** Intel Core i3 (CPU-Only, 8 Logical Cores), 8 GB RAM, Windows 11 Build 26200, Python 3.13.3 AMD64, PyTorch 2.15.0+cpu  
**Architecture Verdict:** **`PASS WITH LIMITATIONS (READY FOR TASK 4 TRAINING)`**

---

## 1. Executive Summary

This report establishes and validates the production-oriented deep recurrent neural network architecture (**`LSTMForecaster`**) for multi-horizon perception reliability forecasting in **PerceptGuard AI**.

Following the successful completion of the Day 9 Task 1 Training Readiness Audit and Day 9 Task 2 Baseline Benchmarking, Task 3 focuses exclusively on the design, mathematical formulation, numerical guardrails, CPU optimization, and training interface compatibility of the forecasting model.

**Crucial Scope Boundary:** In strict adherence to engineering protocol, **NO actual LSTM model training, parameter optimization via backpropagation epochs, or forecasting accuracy evaluations were performed in this task**. All training execution is reserved for subsequent tasks (Task 4: Experiment B).

---

## 2. Readiness Gate Verification

Before finalizing the architecture, all prerequisites were independently audited and verified:

1. **Ground-Truth Reliability Labels Available:**
   - Real annotated labels exist on disk at `datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json` and `frame_reliability_labels.json`.
   - Verified 462 total temporal sequences across $L \in \{5, 10, 15, 20\}$; 138 sequences for $L=5$.
   - Primary reliability target: Frame $F_1$ perception score derived via Hungarian bipartite matching against canonical BDD100K ground-truth bounding boxes at IoU threshold $\tau = 0.50$.
2. **Target Horizon Schema Matching:**
   - Horizons strictly configured as $H1 = 1$, $H3 = 3$, $H5 = 5$ frames ahead ($0.2\text{ s}$, $0.6\text{ s}$, $1.0\text{ s}$ physical lookahead).
   - Target format is strictly float continuous scores in $[0.0, 1.0]$.
3. **Sequence-to-Target Alignment:**
   - Sequences ending at frame $t$ map directly to target frames $t+1, t+3, t+5$.
   - Real Day 5 feature tensors at `datasets/processed/driving_sample/features/sequences/run_official/sequences_L5.npy` ($138 \times 5 \times 27$) match metadata indices 1:1.
4. **Split Integrity & Purging:**
   - Chronological purged split strictly enforces $G=25$ frame embargo gaps between partitions ($N_{\text{train}} = 101$, $N_{\text{val}} = 18$, $N_{\text{test}} = 19$).
   - Zero sequence or frame leakage exists across partitions.
5. **Baseline Benchmark Results Available:**
   - Official baseline benchmarks established in Day 9 Task 2:
     - Test Split Baseline Floor: $\text{MAE} \le 0.000582$ (Historical Mean).
     - Train Split Baseline Floor: $\text{RMSE} < 0.005822$ (Historical Mean).
   - Artifacts stored at `experiments/forecasting/baseline/baseline_benchmark_official/`.

**Readiness Gate Result:** **`PASS`**

---

## 3. Architecture Specification & Mathematical Formulation

### 3.1 Network Topology
```
           Input Sequence Tensor (x)
    Shape: (batch_size, sequence_length, 27)
                       │
                       ▼
        ┌───────────────────────────────┐
        │   Input Validation Guardrails │
        │  (NaN, Inf, Dtype, 3D Check)  │
        └──────────────┬────────────────┘
                       │
                       ▼
        ┌───────────────────────────────┐
        │      Unidirectional LSTM      │
        │   Input: 27  ──> Hidden: 32   │
        │   num_layers: 1 (configurable)│
        │   batch_first: True           │
        └──────────────┬────────────────┘
                       │
                       ▼
        Hidden Representation (h_t = lstm_out[:, -1, :])
             Shape: (batch_size, 32)
                       │
                       ▼
        ┌───────────────────────────────┐
        │    Linear Forecasting Head    │
        │      Linear(32 ──> 3)         │
        └──────────────┬────────────────┘
                       │
                       ▼
        Logits (z = W * h_t + b)
             Shape: (batch_size, 3)
                       │
                       ▼
        ┌───────────────────────────────┐
        │      Sigmoid Activation       │
        │    y_hat = 1 / (1 + e^(-z))   │
        └──────────────┬────────────────┘
                       │
                       ▼
           Multi-Horizon Forecasts
     Shape: (batch_size, 3) ──> [H1, H3, H5]
```

### 3.2 Mathematical Formulation
For each sequence $i$ in batch $B$, with temporal feature steps $x_{i, 1}, \dots, x_{i, L} \in \mathbb{R}^{27}$:
1. **Recurrent State Transitions:**
   $$\begin{aligned}
   f_t &= \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f) \\
   i_t &= \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i) \\
   \tilde{c}_t &= \tanh(W_{xc} x_t + W_{hc} h_{t-1} + b_c) \\
   c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
   o_t &= \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o) \\
   h_t &= o_t \odot \tanh(c_t)
   \end{aligned}$$
   *(Note: Forget gate bias $b_f$ is initialized to $1.0$ to ensure stable long-range gradient propagation).*
2. **Temporal Compression:** The model selects the hidden representation at the final observation step $L$:
   $$h_L = \text{lstm\_out}[:, L-1, :]$$
3. **Multi-Horizon Linear Projection:**
   $$z = h_L W_{\text{head}}^T + b_{\text{head}} \quad \in \mathbb{R}^{B \times 3}$$
4. **Reliability Range Clamping (Sigmoid):**
   $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} \quad \in (0.0, 1.0)^{B \times 3}$$

### 3.3 Horizon Alignment Contract
- `output[:, 0]` $\rightarrow$ Horizon $H1$ ($t+1$ frame ahead)
- `output[:, 1]` $\rightarrow$ Horizon $H3$ ($t+3$ frames ahead)
- `output[:, 2]` $\rightarrow$ Horizon $H5$ ($t+5$ frames ahead)
Mapping verified through programmatic interface `model.get_column_for_horizon(h)`.

---

## 4. Model Input Validation Engine

The `LSTMForecaster.forward()` pass embeds strict validation logic implemented in `src/forecasting/lstm_forecaster.py`:

| Check | Failure Condition | Raised Exception | Action Taken |
| :--- | :--- | :--- | :--- |
| **Tensor Type** | Input is not `torch.Tensor` | `ModelArchitectureError` | Rejection with type error message |
| **Dimensionality** | Input `ndim != 3` | `ModelArchitectureError` | Rejection (must be `(B, L, F)`) |
| **Batch Size** | `batch_size == 0` | `ModelArchitectureError` | Rejection (empty batch not permitted) |
| **Sequence Length** | `seq_len == 0` | `ModelArchitectureError` | Rejection (sequence length must be $\ge 1$) |
| **Feature Count** | `feature_count != input_size` | `ModelArchitectureError` | Rejection with expected vs actual count |
| **Numerical Integrity** | Contains `NaN` or `\pm\infty` | `ModelArchitectureError` | Strict rejection (non-finite protection) |
| **Data Type** | `x.dtype != torch.float32` | Automatic Cast | Safely cast to `torch.float32` without error |

---

## 5. CPU Resource Optimization & Latency Profiling

### 5.1 Host Hardware Constraints
- **Processor:** Intel Core i3 (8 Logical Cores, 2.10 GHz base)
- **Host Memory:** 8.00 GB RAM
- **Operating System:** Windows 11 Build 26200
- **Accelerator:** None (Strictly CPU-only execution; no CUDA GPU)

### 5.2 Model Parameter Footprint
For default configuration ($F=27$, $H=32$, $L=1$, $\text{out}=3$):
- **LSTM Layer 1 Weights & Biases:**
  - $W_{ih}: 4 \times (32 \times 27) = 3,456$ parameters
  - $W_{hh}: 4 \times (32 \times 32) = 4,096$ parameters
  - $b_{ih} + b_{hh}: 4 \times 32 \times 2 = 256$ parameters
  - Subtotal: $7,808$ parameters
- **Linear Forecasting Head:**
  - $W_{\text{head}}: 3 \times 32 = 96$ parameters
  - $b_{\text{head}}: 3$ parameters
  - Subtotal: $99$ parameters
- **Total Trainable Parameters:** **7,907** (0 non-trainable)
- **Weight Memory Footprint:** **30.89 KB** ($7,907 \times 4\text{ bytes} / 1024$)

### 5.3 Empirical CPU Forward-Pass Benchmark
Measured on host Intel Core i3 using 100 benchmark iterations (10 warm-up runs) at batch size $B=8$, sequence length $L=5$:

| Metric | Measured Value |
| :--- | :--- |
| **Mean Forward Latency** | **$0.5405\text{ ms}$** per batch of 8 |
| **Median Forward Latency** | **$0.4939\text{ ms}$** per batch of 8 |
| **Min Forward Latency** | **$0.3560\text{ ms}$** |
| **Max Forward Latency** | **$0.9565\text{ ms}$** |
| **Standard Deviation** | **$0.1246\text{ ms}$** |
| **CPU Forward Throughput** | **$14,802.37\text{ samples / second}$** |
| **Process RSS Memory** | **$186.4\text{ MB}$** ($>97\%$ headroom on 8 GB system) |

*Conclusion:* The architecture is exceptionally lightweight and computationally efficient. A full forward pass across all 138 sequences takes less than $10\text{ ms}$ total on CPU.

---

## 6. Training Interface Compatibility

The architecture is structured to integrate seamlessly with the training pipeline (`src/forecasting/training_engine.py` and `src/forecasting/train_lstm.py`):

1. **Configurable Loss Functions:**
   - Factory function `create_forecasting_loss(loss_name, reduction)` supports:
     - `mse`: Mean Squared Error (`nn.MSELoss`)
     - `mae` / `l1`: Mean Absolute Error (`nn.L1Loss`)
     - `smooth_l1`: Huber loss variant (`nn.SmoothL1Loss`)
   - Decoupled validation function `validate_loss_inputs(predictions, targets)` verifies dimension and finiteness before loss evaluation.
2. **Optimizer Integration:**
   - Fully compatible with `torch.optim.AdamW`, `Adam`, and `SGD`.
   - Verified weight decay and gradient clipping (`torch.nn.utils.clip_grad_norm_`).
3. **Learning Rate Schedulers:**
   - Fully compatible with `torch.optim.lr_scheduler.ReduceLROnPlateau`.
4. **Checkpoint Management:**
   - Atomic saving and loading via `CheckpointManager` verified.
   - Tested full state restoration with zero numerical drift ($\text{atol} = 10^{-7}$).
5. **Deterministic Initialization:**
   - Xavier uniform weight initialization and orthogonal recurrent weights seeded via `torch.manual_seed(seed)` guarantee bit-level reproducible initializations.

---

## 7. Real Artifact Validation

The architecture was validated directly against real dataset artifacts on disk:
- Loaded genuine `datasets/processed/driving_sample/features/sequences/run_official/sequences_L5.npy`.
- Tensor dimensions: `(138, 5, 27)`, float32, 100% finite.
- Forward pass executed on all 138 sequences on CPU.
- Output dimensions: strictly `(138, 3)`.
- All outputs strictly bounded in $(0.0, 1.0)$.
- Integration with `ForecastingDataModule.from_artifacts()` confirmed via `python -m src.forecasting.train_lstm --metadata-file datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json --dry-run` (Exit Code 0).

---

## 8. Automated Testing & Verification

1. **Focused Architecture Tests (`tests/forecasting/test_lstm_forecaster.py`):**
   - 23 unit and integration tests covering:
     - Default initialization and parameter ranges.
     - Parameter bounds checking ($input\_size > 0$, $hidden \ge 4$, $layers \ge 1$, $0 \le dropout < 1$).
     - Variable sequence length processing ($L \in \{5, 10, 15, 20\}$).
     - Multi-sample and variable batch sizes ($B \in \{1, 2, 4, 8, 16, 32\}$).
     - Non-finite rejection (NaN and Inf detection).
     - Non-3D tensor and empty batch rejection.
     - Gradient propagation and backward pass stability.
     - Multi-layer LSTM ($num\_layers = 2$) with dropout ($0.2$).
     - Horizon column lookup interface and error handling.
     - Checkpoint save and load numerical equivalence.
     - Real Day 5 sequence tensor execution.
     - Latency benchmark execution.
     - CLI summary execution.
   - **Result:** **23 / 23 Passed in 15.07s** (100% pass rate).
2. **Full Forecasting Test Suite Regression:**
   - Executed full test suite across all forecasting modules.
   - **Result:** **284 / 284 Passed in 82.35s** (100% pass rate, 0 failures, 0 regressions).

---

## 9. Known Limitations & Task Boundaries

1. **Severe Label Skew (From Day 8):**
   - Over $99.5\%$ of ground-truth reliability targets in this single video sample are $0.0$ due to the container camera rotation.
   - As documented in Day 9 Task 1 and Task 2, an unweighted MSE loss trained on this data will rapidly converge to predicting constant zeros (matching Persistence).
   - *Mitigation for Task 4:* Standard MSE will be evaluated as Experiment B, followed by loss reweighting / asymmetric Huber loss in Experiment C.
2. **Single Camera Perspective:**
   - Architecture is validated on BDD100K sample `026c7465-309f6d33.mp4`. Generalization across multiple camera angles requires multi-video training datasets.
3. **Task Scope Boundaries:**
   - No model training was executed in Task 3.
   - No accuracy improvements are claimed.
   - Weights remain in initial un-trained state.

---

## 10. Training Readiness Determination

| Criterion | Required Standard | Observed Status | Audit Determination |
| :--- | :--- | :--- | :---: |
| **Architecture Implementation** | Modular PyTorch LSTM for H1, H3, H5 | Implemented in `lstm_forecaster.py` | **`MET`** |
| **Input Validation Engine** | Rejects non-3D, NaN, Inf, empty, feature mismatch | Validated in 23 focused tests | **`MET`** |
| **CPU Resource Optimization** | Bounded footprint, CPU-only, $< 500\text{ MB}$ RSS | 7,907 params, $30.89\text{ KB}$, $0.54\text{ ms}$ | **`MET`** |
| **Real Artifact Compatibility** | Compatible with `sequences_L5.npy` and labeled metadata | Verified via real tensor execution & dry-run | **`MET`** |
| **Training Interface** | Loss, optimizer, checkpoint, scheduler hooks | Verified and tested | **`MET`** |
| **Test Coverage** | 100% passing tests, 0 regressions | 23/23 focused, 284/284 suite passed | **`MET`** |

**Final Architecture Verdict:** **`PASS WITH LIMITATIONS (READY FOR TASK 4 SUPERVISED TRAINING)`**

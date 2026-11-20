# PerceptGuard AI — Advanced LSTM Forecasting Architecture

## 1. Overview & Architecture Design

PerceptGuard AI forecasts future perception reliability across upcoming video frames.
Building upon Day 5's temporal feature sequence preparation, Day 8's ground-truth reliability label generation, and Day 9 Tasks 1–2 baseline benchmarks, Day 9 Task 3 establishes and formally certifies the primary deep learning recurrent model: **`LSTMForecaster`**.

The architecture processes historical temporal feature sequences to capture perception degradation dynamics, tracking stability trends, and visual quality degradation slopes, predicting multi-horizon reliability scores ($t+1, t+3, t+5$ frames ahead).

### Architectural Topology
```
           Input Sequence Tensor
    (batch_size, sequence_length, feature_count)
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Input Validation Guardrails │
        │  (NaN, Inf, Dtype, 3D Check)  │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │     Unidirectional LSTM       │
        │   (hidden_size=32, layers=1)  │
        │        batch_first=True       │
        └───────────────┬───────────────┘
                        │
                        ▼
        Final Temporal Representation (h_t)
              (batch_size, hidden_size)
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Linear Forecasting Head     │
        │      Linear(32 -> 3)          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │      Sigmoid Activation       │
        │  (Output Range: [0.0, 1.0])   │
        └───────────────┬───────────────┘
                        │
                        ▼
             Multi-Horizon Forecast
        (batch_size, 3) -> [H1, H3, H5]
```

---

## 2. Input Tensor Contract

- **Tensor Dimensionality:** Strictly 3-dimensional (`ndim == 3`).
- **Shape Specification:** `(batch_size, sequence_length, feature_count)` $\rightarrow (B, L, 27)$.
- **Batch Dimension:** Configurable ($B \ge 1$; e.g. 1, 4, 8, 16, 32).
- **Temporal Sequence Length ($L$):** Variable window lengths supported ($L \in \{5, 10, 15, 20\}$).
- **Feature Dimension ($F$):** Dynamically resolved from `feature_manifest.json` ($F=27$ features: 6 detection, 8 motion, 13 visual quality).
- **Data Type & Numerical Guardrails:** `torch.float32`. Strictly finite values (strict rejection of NaNs and $\pm\infty$ via `ModelArchitectureError`).

---

## 3. Output Tensor Contract & Horizon Mapping

- **Tensor Dimensionality:** Strictly 2-dimensional (`ndim == 2`).
- **Shape Specification:** `(batch_size, number_of_horizons)` $\rightarrow (B, 3)$.
- **Data Type:** `torch.float32`.
- **Target Value Domain:** Continuous reliability score in the bounded range $[0.0, 1.0]$.
  - `0.0`: Total perception failure (zero bounding box recall, complete sensor dropout).
  - `1.0`: Optimal perception reliability (accurate detection, high IoU against ground truth).

### Column-to-Horizon Ordering
Output tensor columns strictly preserve configured horizon offsets:
- **Column 0:** Forecast Horizon $h = 1$ ($t+1$ frame ahead; $+0.2\text{ s}$)
- **Column 1:** Forecast Horizon $h = 3$ ($t+3$ frames ahead; $+0.6\text{ s}$)
- **Column 2:** Forecast Horizon $h = 5$ ($t+5$ frames ahead; $+1.0\text{ s}$)

The mapping is explicitly exposed via `model.get_column_for_horizon(horizon: int) -> int` and `model.horizon_to_index`.

---

## 4. Target Contract & Ground-Truth Readiness Policy

1. **Continuous Regression Semantics:**
   - The primary target represents continuous future object detection reliability ($F_1$ score derived from Hungarian bipartite matching against canonical BDD100K annotations at IoU threshold $\tau = 0.50$).
2. **Ground-Truth Readiness Status:**
   - Verified Day 8 ground-truth labels exist at `datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json`.
   - Chronological purged split with $G=25$ embargo gap guarantees zero temporal leakage between Train ($N=101$), Val ($N=18$), and Test ($N=19$) partitions.
3. **Scientific Integrity Gate:**
   - In Task 3, only architecture validation, numerical verification, and CPU latency benchmarks are performed.
   - **No model training epochs or accuracy evaluations were executed in this task.** Actual model training begins in Task 4 (Experiment B).

---

## 5. Model Configuration Options

Configured via `config/forecasting_config.yaml` and strongly typed `ModelConfig`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `architecture` | `str` | `"lstm"` | Recurrent model architecture |
| `input_size` | `str` / `int` | `"auto"` | Feature count (auto-resolved from manifest to 27) |
| `hidden_size` | `int` | `32` | LSTM hidden state dimension ($\ge 4$) |
| `num_layers` | `int` | `1` | Number of stacked recurrent layers ($\ge 1$) |
| `dropout` | `float` | `0.0` | Dropout probability in $[0.0, 1.0)$ |
| `output_horizons` | `list[int]` | `[1, 3, 5]` | Target forecast frame horizons |
| `output_size` | `int` | `3` | Dimension matching length of `output_horizons` |
| `target_type` | `str` | `"regression"` | Continuous perception reliability |
| `activation` | `str` | `"sigmoid"` | Output activation (`sigmoid`, `none`, or `identity`) |
| `batch_first` | `bool` | `True` | Expected input shape format `(B, L, F)` |

---

## 6. Model Factory & Dynamic Resolution

Implemented in `src/forecasting/model_factory.py`:
- **`build_forecasting_model(config, input_size, artifacts_dir, target_contract, device)`**:
  - Dynamically resolves `input_size` from `feature_manifest.json` if set to `"auto"`.
  - Rejects unsupported architectures (`"gru"` or `"transformer"` are blocked).
  - Validates output horizons against `ModelOutputSpec`.
  - Places the model strictly on the configured CPU device.

---

## 7. Decoupled Loss Interface

Implemented in `src/forecasting/lstm_forecaster.py`:
- **`create_forecasting_loss(loss_name="mse", reduction="mean")`**:
  - `"mse"`: `nn.MSELoss` (Mean Squared Error for continuous reliability).
  - `"mae"` / `"l1"`: `nn.L1Loss` (Mean Absolute Error for robust evaluation).
  - `"smooth_l1"`: `nn.SmoothL1Loss` (Huber loss variant).
- **`validate_loss_inputs(predictions, targets)`**:
  - Validates tensor shape alignment, finite numbers, and float32 dtypes.
  - Keeps loss computation completely decoupled from the model forward pass.

---

## 8. Model Parameter & Memory Analysis

Parameter calculation for $F=27$, $H=32$, $L=1$, $\text{out}=3$:
- **LSTM Input-to-Hidden Weights:** $4 \times (32 \times 27) = 3,456$
- **LSTM Hidden-to-Hidden Weights:** $4 \times (32 \times 32) = 4,096$
- **LSTM Biases:** $4 \times 32 \times 2 = 256$
- **Linear Head Weights:** $3 \times 32 = 96$
- **Linear Head Biases:** $3$
- **Total Trainable Parameters:** **7,907**
- **Non-Trainable Parameters:** **0**
- **Estimated Parameter Memory:** **30.89 KB** (at 4 bytes / float32)

---

## 9. CPU Execution & Latency Benchmarks

Empirically measured on host Intel Core i3 CPU (8 logical cores, PyTorch CPU threads: 2):

| Metric | Batch Size = 8, Seq Length = 5 |
|---|---|
| **Mean Forward Latency** | **$0.5405\text{ ms}$** |
| **Median Forward Latency** | **$0.4939\text{ ms}$** |
| **Min Latency** | **$0.3560\text{ ms}$** |
| **Max Latency** | **$0.9565\text{ ms}$** |
| **Model Throughput** | **$14,802.37\text{ samples / sec}$** |
| **Parameter Memory** | **$30.89\text{ KB}$** |
| **Process RSS Memory** | **$186.4\text{ MB}$** |

---

## 10. CLI Usage

### Inspect Model Architecture & Summary
```bash
python -m src.forecasting.lstm_forecaster --summary
```

### Run Model CPU Latency Benchmark
```bash
python -m src.forecasting.lstm_forecaster --summary --benchmark
```

### Test Data & Architecture Compatibility via Dry Run
```bash
python -m src.forecasting.train_lstm --metadata-file datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json --dry-run
```

---

## 11. Test Coverage & Verification

Automated test suite in `tests/forecasting/`:
- `test_lstm_forecaster.py`: 23 unit tests covering layer instantiation, input validation, output shapes, clipping, backpropagation, checkpoint save/load, real sequence tensors, and determinism.
- `test_model_factory.py`: 12 unit tests covering dynamic input dimension resolution, contract checks, and architecture blocking.
- `test_model_output_contract.py`: 9 integration tests verifying real Day 5 tensor compatibility ($L \in \{5, 10, 15, 20\}$) and multi-horizon mappings.
- **Full Forecasting Test Suite:** **284 / 284 Passed (100% Pass Rate)**.

---

## 12. Known Limitations & Task Scope Boundaries

1. **Severe Label Imbalance:** Over $99.5\%$ of ground-truth reliability targets are $0.0$ due to camera rotation container mismatch. Standard unweighted MSE (Task 4) will establish the unweighted baseline; asymmetric loss reweighting (Task 5) is required to optimize transient detection.
2. **Single Video Domain:** Current artifacts are from video `026c7465-309f6d33.mp4`.
3. **No Training in Task 3:** Model parameters remain strictly un-trained. No performance claims or loss curves are generated.

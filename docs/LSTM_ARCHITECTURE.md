# PerceptGuard AI — Advanced LSTM Forecasting Architecture

## 1. Overview & Architecture Design

PerceptGuard AI forecasts future perception reliability across upcoming video frames.
Building upon Day 5's temporal feature sequence preparation and Day 6 Tasks 1–2 baseline foundations, Day 6 Task 3 establishes the primary deep learning recurrent model: **`LSTMForecaster`**.

The architecture processes historical temporal feature sequences to capture perception degradation dynamics, tracking stability trends, and visual quality degradation slopes, predicting multi-horizon reliability scores ($t+1, t+3, t+5$ frames ahead).

### Architectural Topology
```
           Input Sequence Tensor
   (batch_size, sequence_length, feature_count)
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
       │     Linear(32 -> 3)           │
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
       (batch_size, 3) -> [h1, h3, h5]
```

---

## 2. Input Tensor Contract

- **Tensor Dimensionality:** Strictly 3-dimensional (`ndim == 3`).
- **Shape Specification:** `(batch_size, sequence_length, feature_count)`.
- **Batch Dimension:** Configurable ($B \ge 1$; e.g. 1, 4, 8, 16).
- **Temporal Sequence Length ($L$):** Variable window lengths supported ($L \in \{5, 10, 15, 20\}$).
- **Feature Dimension ($F$):** Dynamically resolved from `feature_manifest.json` ($F=27$ for Day 5 artifacts).
- **Data Type & Numerical Guardrails:** `torch.float32`. Strictly finite values (rejection of NaNs and $\pm\infty$).

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
- **Column 0:** Forecast Horizon $h = 1$ ($t+1$ frame ahead)
- **Column 1:** Forecast Horizon $h = 3$ ($t+3$ frames ahead)
- **Column 2:** Forecast Horizon $h = 5$ ($t+5$ frames ahead)

The mapping is explicitly exposed via `model.get_column_for_horizon(horizon: int) -> int` and `model.horizon_to_index`.

---

## 4. Target Contract & Ground-Truth Readiness Policy

1. **Continuous Regression Semantics:**
   - The primary target represents continuous future object detection reliability (e.g., ground-truth bounding box IoU / recall under Hungarian bipartite matching).
2. **Scientific Integrity Gate (CRITICAL):**
   - Day 5 sequence metadata currently records `target_label_available: false` and `target_label: null`.
   - YOLO detection confidence alone is **NOT** ground-truth perception reliability.
   - Supervised training readiness is honestly marked as **`NOT AVAILABLE`** for real data.
   - Forward pass, dimension checks, gradient propagation, and unit tests execute using synthetic fixtures and actual unlabelled Day 5 tensors. No deceptive accuracy metrics are claimed without true ground-truth annotations.

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
  - Rejects unsupported architectures (`"gru"` or `"transformer"` are blocked in Task 3).
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

Measured on host Intel Core i3 CPU (8 logical threads, PyTorch CPU threads: 2):

| Metric | Batch Size = 8, Seq Length = 5 |
|---|---|
| **Mean Forward Latency** | **0.5872 ms** |
| **Median Forward Latency** | **0.5405 ms** |
| **Min Latency** | **0.4378 ms** |
| **Max Latency** | **1.2400 ms** |
| **Model Throughput** | **13,624 samples / sec** |
| **Parameter Memory** | **30.89 KB** |

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

---

## 11. Test Coverage & Verification

Automated test suite in `tests/forecasting/`:
- `test_lstm_forecaster.py`: 20 unit tests covering layer instantiation, input validation, output shapes, clipping, backpropagation, and determinism.
- `test_model_factory.py`: 11 unit tests covering dynamic input dimension resolution, contract checks, and GRU blocking.
- `test_model_output_contract.py`: 9 integration tests verifying real Day 5 tensor compatibility ($L \in \{5, 10, 15, 20\}$) and multi-horizon mappings.
- **Task 3 Test Subtotal:** **40 / 40 Passed (100% Pass Rate)**.

---

## 12. Known Limitations & Task Scope Boundaries

1. **Training Pipeline Unimplemented:** Model training loops, optimizers, learning rate schedulers, and backpropagation epochs belong to Day 6 Task 5.
2. **GRU Architecture Excluded:** Day 6 Task 3 implements only LSTM. GRU variants are reserved for subsequent architectural comparisons.
3. **Supervised Target Unavailability:** Supervised real-world reliability training remains `NOT AVAILABLE` until true ground-truth labels are ingested.

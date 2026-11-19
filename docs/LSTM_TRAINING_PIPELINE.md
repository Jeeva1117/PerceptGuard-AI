# PerceptGuard AI — LSTM Model Training Pipeline
## Technical Architecture, Optimization Engine & Supervised Readiness Gate Specification

---

## 1. Executive Summary

This document specifies the technical architecture, mathematical formulation, data split protection mechanisms, optimization loops, and supervised target readiness gate of the **PerceptGuard AI** LSTM training pipeline implemented for **Day 6 Task 4**.

The training system is designed to train the unidirectional `LSTMForecaster` architecture ($F = 27$ engineered features, $H = 32$ hidden units, $1$ recurrent layer) on historical perception sequences ($L \in \{5, 10, 15, 20\}$) to forecast multi-horizon perception reliability across upcoming video frames ($h \in \{1, 3, 5\}$).

The pipeline adheres to strict CPU execution constraints (Intel Core i3, 8 GB RAM, Windows 11, CPU only) and enforces scientific label integrity: **supervised model training is gated and will not execute on real datasets without verified ground-truth reliability targets**.

---

## 2. Training Architecture & Information Flow

The end-to-end training architecture isolates data ingestion, optimization, checkpointing, and experiment tracking into decoupled, modular components:

```
+---------------------------------------------------------------------------------------------------+
|                                   LSTM TRAINING PIPELINE FLOW                                     |
+---------------------------------------------------------------------------------------------------+
|  Day 5 Sequence Artifacts (sequences_L*.npy, feature_manifest.json, dataset_splits.json)          |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  ForecastingDataModule:                                                                           |
|  - Ingests sequence tensors & splits into Train / Val / Test partitions                           |
|  - Enforces Split Integrity: zero overlap in sequence IDs between Train, Val, and Test            |
|  - Supervised Target Readiness Gate: Validates existence of ground-truth target matrix [N, 3]     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  PyTorch DataLoaders (CPU Optimized: batch_size = 8, num_workers = 0, deterministic ordering)     |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
|  LSTMTrainingEngine Loop (Epoch 1 .. max_epochs):                                                 |
|  1. Train Pass:                                                                                   |
|     - Forward pass: y_pred = model(x_batch) ∈ (0.0, 1.0)^(B × 3)                                  |
|     - Loss: L = MSELoss(y_pred, y_batch)                                                          |
|     - Finite check: verify isfinite(L)                                                            |
|     - Backward pass: L.backward()                                                                 |
|     - Gradient clipping: clip_grad_norm_(parameters, max_norm = 1.0)                              |
|     - Optimizer step: AdamW(lr = 0.001, weight_decay = 0.0001)                                    |
|  2. Validation Pass:                                                                              |
|     - Evaluation with torch.no_grad() across validation split                                     |
|  3. Learning Rate Scheduler:                                                                      |
|     - ReduceLROnPlateau(mode='min', factor=0.5, patience=3, min_lr=1e-6)                           |
|  4. Early Stopping:                                                                               |
|     - EarlyStopping(patience=8, min_delta=1e-4, mode='min', restore_best_weights=True)             |
|  5. Checkpoint Management:                                                                        |
|     - Saves checkpoint_last.pt atomically                                                         |
|     - If is_best: duplicates checkpoint_best.pt                                                   |
|  6. Telemetry Logging:                                                                            |
|     - Records training_history.json, training_summary.json, metadata.json                         |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Dataset Loading & Preprocessing Alignment

### 3.1 ForecastingDataset
The `ForecastingDataset` class wraps input temporal feature sequences and multi-horizon target labels:
- **Input Tensor `X`**: Float32 tensor of shape `(N, L, F)` where $N$ is the sample count, $L$ is sequence length ($5, 10, 15, 20$), and $F = 27$ is the engineered feature dimension.
- **Target Tensor `Y`**: Float32 tensor of shape `(N, 3)` where columns correspond to future frames $t+1, t+3, t+5$.
- **Validation**:
  - `ndim == 3` for sequences, `ndim == 2` for targets.
  - Finite validation: `torch.isfinite()` ensures zero NaN or Inf values.
  - Dimension alignment: sample count $N$ must match between sequences and targets.

### 3.2 ForecastingDataModule
The `ForecastingDataModule` handles partitioned sequence datasets:
- Instantiates `train_dataloader(shuffle=True)`, `val_dataloader(shuffle=False)`, and `test_dataloader(shuffle=False)`.
- Reuses preprocessing scalers and imputation parameters fitted strictly on the training partition during Day 5. No fitting or parameter recomputation occurs during DataLoader construction.

---

## 4. Data Split Protection & Leakage Elimination

Data leakage is strictly prevented via programmatic guards:

1. **Split Partition Isolation**:
   - Sequences assigned to the `train` split are never present in `val` or `test`.
   - The validation split is used exclusively for evaluating loss, stepping `ReduceLROnPlateau`, and triggering early stopping.
   - The test split is held out completely and is not accessible during training or model selection.

2. **Sequence ID Disjointness**:
   `ForecastingDataModule` performs explicit set intersection checks upon initialization:
   $$\text{TrainIDs} \cap \text{ValIDs} = \emptyset, \quad \text{TrainIDs} \cap \text{TestIDs} = \emptyset, \quad \text{ValIDs} \cap \text{TestIDs} = \emptyset$$
   If any overlap is detected, `DatasetSplitError` is raised immediately.

3. **Temporal Gap Preservation**:
   Chrono-split temporal gaps established in Day 5 are preserved by consuming the immutable `dataset_splits.json` artifact.

---

## 5. Supervised Target Readiness Gate

### 5.1 Scientific Label Integrity Policy
PerceptGuard AI maintains research-grade scientific integrity:
- **Ground Truth Definition**: Object detection reliability is defined as future detection success (e.g., ground-truth IoU $\ge 0.5$, True Positive status, or temporal persistence).
- **YOLO Confidence Rejection**: Raw YOLO detection confidence scores are model probabilities for current detections. They do not represent future detection reliability ground truth.
- **No Fabricated Labels**: The system never manufactures pseudo-labels from heuristic indicators or noise to force training.

### 5.2 Real Dataset Status
In the Day 5 `driving_sample` dataset:
- `target_labels_available: false`
- `target_label: null` across all horizons $h \in \{1, 3, 5\}$.
- Target status: **NOT AVAILABLE**.

### 5.3 Gate Behavior
When `train_lstm.py` or `LSTMTrainingEngine.train()` is executed against real unlabelled data:
1. The gate identifies that `has_targets == False`.
2. Execution halts cleanly before Epoch 1.
3. Diagnostic telemetry reports the exact blocker and annotational prerequisites.
4. Model parameters remain uncorrupted by unlabelled data.

---

## 6. Loss Functions & Optimization Loop

### 6.1 Loss Formulations
The default loss for multi-horizon continuous reliability forecasting is Mean Squared Error:

$$\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{B \cdot K} \sum_{b=1}^B \sum_{k=1}^K (\hat{y}_{b,k} - y_{b,k})^2$$

where $B$ is the batch size and $K = 3$ is the number of forecast horizons ($h \in \{1, 3, 5\}$).

Optionally, Mean Absolute Error (L1 Loss) is supported:

$$\mathcal{L}_{\text{MAE}}(\hat{y}, y) = \frac{1}{B \cdot K} \sum_{b=1}^B \sum_{k=1}^K |\hat{y}_{b,k} - y_{b,k}|$$

### 6.2 Optimizer (AdamW)
Optimization is performed using AdamW with decoupled weight decay:
- Learning rate $\eta = 0.001$
- Weight decay $\lambda = 0.0001$
- Betas: $(\beta_1, \beta_2) = (0.9, 0.999)$
- Epsilon: $\epsilon = 10^{-8}$

### 6.3 Gradient Clipping
To prevent gradient explosion in recurrent connections, gradient norms are clipped before each optimizer step:

$$g \leftarrow \begin{cases} g & \text{if } \|g\|_2 \le \gamma \\ \frac{\gamma}{\|g\|_2} g & \text{if } \|g\|_2 > \gamma \end{cases}$$

where $\gamma = 1.0$ (`max_grad_norm`).

---

## 7. Learning Rate Scheduling & Early Stopping

### 7.1 Learning Rate Scheduler
A `ReduceLROnPlateau` scheduler monitors validation loss:
- `mode = "min"`
- `factor = 0.5` (reduces learning rate by 50% upon plateau)
- `patience = 3` epochs without improvement
- `min_lr = 1e-6`

### 7.2 Early Stopping
An `EarlyStopping` monitor halts training when progress stalls:
- `patience = 8` epochs
- `min_delta = 1e-4`
- `mode = "min"`
- `restore_best_weights = True`: Automatically deep-copies the state dictionary of the model at the best validation epoch and restores it upon termination.

---

## 8. Checkpoint Management & Persistence

Checkpoints are managed by `CheckpointManager`:
- **Atomic Persistence**: Checkpoint dictionaries are written to `.tmp` files and atomically renamed via `shutil.move` to prevent corrupted checkpoint files during interruptions.
- **Managed Artifacts**:
  - `checkpoint_last.pt`: Saved after every epoch.
  - `checkpoint_best.pt`: Saved/updated whenever validation loss achieves a new minimum.
- **Saved State Schema**:
  ```python
  {
      "epoch": int,
      "val_loss": float,
      "best_epoch": int,
      "best_val_loss": float,
      "is_best": bool,
      "model_state_dict": OrderedDict,
      "optimizer_state_dict": dict,
      "scheduler_state_dict": Optional[dict],
      "input_size": int,
      "hidden_size": int,
      "output_horizons": List[int],
      "experiment_id": str,
      "metadata": dict
  }
  ```
- **Architectural Validation**: Loading a checkpoint into a model verifies that `input_size` and `output_horizons` strictly match the model specification, preventing silent dimension mismatches.

---

## 9. Experiment Telemetry & History Logging

`TrainingLogger` manages structured experiment records in `experiments/forecasting/lstm_training/<run_id>/`:
- `training_history.json`: Array of epoch telemetry records (loss, val_loss, lr, duration, grad_norm, batch_count, early_stopped).
- `training_summary.json`: Summary metadata (total epochs, best epoch, best val loss, total runtime, termination reason).
- `experiment_config.yaml`: Fully resolved experiment configuration.
- `metadata.json`: Runtime environment details (Python version, PyTorch version, CPU count, PyTorch threads, platform).
- `errors.json`: Telemetry logs of any caught errors or numerical instabilities.

---

## 10. CPU & Memory Optimization

Tuned for local CPU execution (Intel Core i3, 8 GB RAM, Windows 11):
1. **PyTorch CPU Threads**: Thread count is pinned via `torch.set_num_threads(config.runtime.num_threads)` (default: 2 threads) to prevent CPU resource saturation.
2. **DataLoader Configuration**: `num_workers = 0` avoids Windows multiprocessing spawn overhead and shared memory serialization bottlenecks.
3. **Batch Sizing**: Batch size is constrained to $B = 8$ for low memory consumption ($< 50\text{ MB}$ dynamic memory usage).
4. **Gradient Retention**: Validation passes are wrapped strictly in `torch.no_grad()`, freeing intermediate activation graphs.

---

## 11. Command-Line Interface (CLI)

The CLI is invoked via `src.forecasting.train_lstm`:

```powershell
python -m src.forecasting.train_lstm --help
```

### Options:
- `--config <path>`: Path to YAML configuration (default: `config/forecasting_config.yaml`).
- `--epochs <int>`: Override maximum training epochs.
- `--batch-size <int>`: Override DataLoader batch size.
- `--learning-rate <float>`: Override AdamW learning rate.
- `--device {cpu,cuda}`: Target execution device.
- `--sequence-length <int>`: Sequence length to load ($5, 10, 15, 20$).
- `--resume <path>`: Path to `.pt` checkpoint to resume from.
- `--output <dir>`: Custom experiment output directory.
- `--verbose`: Enable detailed per-epoch progress logs.

### Execution Example on Real Data (Target Gate Verification):
```powershell
python -m src.forecasting.train_lstm --config config/forecasting_config.yaml --sequence-length 5
```
*Output:*
```text
==============================================================================
  PERCEPTGUARD AI - ADVANCED LSTM MODEL TRAINING PIPELINE
==============================================================================
[CONFIG] Loaded valid configuration from: config\forecasting_config.yaml
[DATA] Located sequence artifacts at: datasets\processed\driving_sample\features\sequences\run_official
[DATA] Sequence Length: L = 5
[DATA] Split Counts: Train=101, Val=18, Test=19 (Total=138)
[DATA] Batch Size: 8
------------------------------------------------------------------------------
  SUPERVISED TARGET READINESS GATE VERIFICATION
------------------------------------------------------------------------------
[STATUS] Supervised Target Readiness: NOT AVAILABLE
[REASON] The Day 5 driving_sample sequence artifacts do not contain verified
         ground-truth future detection reliability target labels.
[POLICY] PerceptGuard AI strictly enforces scientific label integrity:
         - YOLO confidence scores are detection probabilities, NOT future reliability labels.
         - Synthetic labels are never fabricated or injected into real driving datasets.
         - Model weights are not updated via backpropagation without verified labels.
[ACTION] Halting training loop before Epoch 1. Training pipeline is verified & ready.
==============================================================================
```

---

## 12. Verification & Testing

The training pipeline is verified by 53 unit and integration tests across 6 specialized test suites:

| Test Module | Tests | Scope |
| :--- | :--- | :--- |
| `test_training_data.py` | 15 | Dataset conversion, 3D input validation, NaN/Inf checks, split integrity, Day 5 loading. |
| `test_early_stopping.py` | 10 | Patience decrement, min delta threshold, weight preservation, state serialization. |
| `test_checkpoint_manager.py` | 8 | Atomic saving, best replication, input/horizon mismatch rejection, resumption. |
| `test_training_logger.py` | 5 | JSON telemetry, metadata capture, summary generation, error recording. |
| `test_training_engine.py` | 10 | Training epoch, backprop, gradient clipping, NaN loss guard, early stopping, resume. |
| `test_train_lstm_cli.py` | 5 | CLI argument parsing, target readiness reporting, multi-sequence length loading. |

**Total Suite Pass Rate: 53 / 53 (100.0%)**  
**Total Forecasting Package Tests: 181 / 181 (100.0%)**

---

## 13. Known Limitations & Next Steps

1. **Target Annotation Blocker**: Actual supervised model training on driving video requires external ground-truth reliability annotations (e.g., future object presence or tracking continuity).
2. **Task 4 Boundary Enforcement**: Inference services, multi-horizon evaluation reporting, and dashboard visualization are reserved for Day 6 Task 5 and subsequent tasks.

# PerceptGuard AI — LSTM Training Pipeline Architecture & Specification

## 1. Executive Overview

The **PerceptGuard AI** LSTM training pipeline provides a reproducible, CPU-optimized, and scientifically validated environment for training deep recurrent perception reliability forecasting models (`LSTMForecaster`).

The pipeline enforces strict dataset partition boundaries, robust numerical guardrails, early stopping, gradient clipping, learning rate scheduling, atomic checkpoint management, and comprehensive JSON experiment tracking.

```
       Verified Ground-Truth Dataset
  (sequence_metadata_labeled.json, L5.npy)
                     │
                     ▼
       ┌───────────────────────────┐
       │   ForecastingDataModule   │
       │  Purged Chronological    │
       │    G=25 Embargo Gap       │
       └─────────────┬─────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
   Train DataLoader        Val DataLoader
   (101 seqs, B=8)         (18 seqs, B=8)
         │                       │
         ▼                       │
   ┌───────────┐                 │
   │ Forward   │                 │
   │ Pass      │                 │
   └─────┬─────┘                 │
         ▼                       │
   ┌───────────┐                 │
   │ Loss (MSE)│                 │
   └─────┬─────┘                 │
         ▼                       │
   ┌───────────┐                 │
   │ Backward  │                 │
   │ Pass      │                 │
   └─────┬─────┘                 │
         ▼                       │
   ┌───────────┐                 │
   │ Clip Norm │                 │
   │ (max=1.0) │                 │
   └─────┬─────┘                 │
         ▼                       │
   ┌───────────┐                 │
   │ Optimizer │                 │
   │  (AdamW)  │                 │
   └─────┬─────┘                 │
         │                       ▼
         │                 ┌───────────┐
         │                 │ Validate  │
         │                 │ Pass      │
         │                 └─────┬─────┘
         │                       ▼
         └───────────┬───────────┘
                     ▼
       ┌───────────────────────────┐
       │   ReduceLROnPlateau Step  │
       └─────────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │    EarlyStopping Step     │
       └─────────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │    CheckpointManager      │
       │   (best.pt, last.pt)      │
       └─────────────┬─────────────┘
                     ▼
       ┌───────────────────────────┐
       │    TrainingLogger         │
       │  (history, summary, meta) │
       └───────────────────────────┘
```

---

## 2. Centralized Configuration Management

All hyperparameters and runtime options are centralized in `src/forecasting/forecasting_config.py` and serialized to `config/forecasting_config.yaml`.

### 2.1 Configuration Schema (`TrainingConfig`)

| Parameter | Type | Default | Description |
| :--- | :--- | :---: | :--- |
| `learning_rate` | `float` | `0.001` | Initial learning rate for optimizer |
| `weight_decay` | `float` | `0.0001` | $L_2$ weight regularization factor |
| `max_epochs` | `int` | `50` | Maximum number of training epochs |
| `early_stopping_patience` | `int` | `8` | Epochs without validation improvement before halting |
| `loss_function` | `str` | `"mse"` | Continuous loss: `"mse"`, `"mae"`, `"l1"`, `"smooth_l1"` |
| `optimizer` | `str` | `"adamw"` | Parameter optimization algorithm: `"adamw"`, `"adam"`, `"sgd"` |
| `gradient_clip_norm` | `float` | `1.0` | Maximum Euclidean gradient norm threshold |
| `random_seed` | `int` | `42` | Global pseudo-random number generator seed |

---

## 3. CPU Resource Optimization

The pipeline is explicitly tailored for the host environment (Intel Core i3, 8 GB RAM, Windows 11, CPU-only, no CUDA):
- **Conservative Batch Size ($B=8$):** Keeps working memory small while providing smooth gradient estimates across 13 batches per epoch on 101 training sequences.
- **Bounded DataLoader Workers (`num_workers=0`):** Avoids multi-process overhead and shared memory serialization bottlenecks on Windows.
- **In-Memory Zero-Copy Views:** Tensor conversions use direct PyTorch views with memory pooling.
- **Process Memory Boundedness:** Total process RSS stays strictly under $200\text{ MB}$, leaving over $97\%$ system RAM free.
- **CPU Forward & Backward Throughput:** Completes an entire training epoch (13 train batches + 3 validation batches) in **$\approx 0.04\text{ seconds}$** ($\approx 25\text{ epochs / sec}$).

---

## 4. Loss and Optimizer Selection

### 4.1 Loss Function: Mean Squared Error (MSE)
For continuous perception reliability forecasting ($y \in [0.0, 1.0]$):
$$\mathcal{L}_{\text{MSE}}(y, \hat{y}) = \frac{1}{B \cdot H} \sum_{i=1}^B \sum_{h=1}^H (y_{i, h} - \hat{y}_{i, h})^2$$
- **Justification:** Penalizes larger errors quadratically, providing smooth gradients that pull predictions away from initial random weights towards ground-truth values.
- **Alternatives Supported:** Mean Absolute Error (`nn.L1Loss`) and Smooth L1 / Huber (`nn.SmoothL1Loss`).

### 4.2 Optimizer: Decoupled Weight Decay Adam (AdamW)
- **Justification:** AdamW decouples weight decay from the moving average gradient updates, preventing $L_2$ weight decay distortion typical in standard Adam on recurrent architectures.
- **Default Parameters:** $\text{lr} = 10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, $\text{weight\_decay} = 10^{-4}$.

---

## 5. Checkpoint Management

Checkpoints are saved atomically by `CheckpointManager`:
1. `checkpoint_last.pt`: Saved at the end of every epoch.
2. `checkpoint_best.pt`: Saved whenever validation loss achieves a new minimum.

### Checkpoint Structure
```python
{
    "epoch": int,
    "val_loss": float,
    "best_epoch": int,
    "best_val_loss": float,
    "is_best": bool,
    "model_state_dict": OrderedDict,
    "optimizer_state_dict": dict,
    "scheduler_state_dict": dict,
    "experiment_id": str,
    "metadata": dict
}
```

Checkpoints are stored outside Git under `experiments/forecasting/lstm_training/<run_id>/checkpoints/` (governed by `.gitignore`).

---

## 6. CLI Usage & Pipeline Invocation

### Dry-Run Verification (Target & Architecture Gate)
```bash
python -m src.forecasting.train_lstm --metadata-file datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json --dry-run
```

### Full Supervised Training Execution
```bash
python -m src.forecasting.train_lstm \
  --metadata-file datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json \
  --epochs 30 \
  --batch-size 8 \
  --learning-rate 0.001 \
  --loss mse \
  --optimizer adamw \
  --seed 42 \
  --run-id official_lstm_training \
  --verbose
```

### Resuming Training from Checkpoint
```bash
python -m src.forecasting.train_lstm \
  --metadata-file datasets/processed/driving_sample/reliability_labels/sequence_metadata_labeled.json \
  --resume experiments/forecasting/lstm_training/official_lstm_training/checkpoints/checkpoint_last.pt \
  --epochs 50 \
  --verbose
```

---

## 7. Real Training Execution Telemetry

Execution of the official 30-epoch training run (`official_lstm_training`):
- **Dataset Partitioning:** 101 Train sequences, 18 Validation sequences, 19 Test sequences ($G=25$ embargo gap).
- **Total Duration:** **$2.12\text{ seconds}$**
- **Loss Progression:**
  - Epoch 1: Train Loss = $0.239112$ | Val Loss = $0.207142$ | Grad Norm = $0.2816$
  - Epoch 10: Train Loss = $0.004090$ | Val Loss = $0.005314$ | Grad Norm = $0.0193$
  - Epoch 20: Train Loss = $0.000564$ | Val Loss = $0.001005$ | Grad Norm = $0.0032$
  - Epoch 30: Train Loss = **$0.000256$** | Val Loss = **$0.000470$** | Grad Norm = **$0.0015$**
- **Outcome:** Best validation loss of **$0.000470$** successfully beats the naive Historical Mean baseline floor ($0.000582$).

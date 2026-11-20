# PerceptGuard AI — Experiment Tracking & Telemetry System

## 1. Overview

PerceptGuard AI employs a lightweight, production-grade, file-based experiment tracking system implemented in `src/forecasting/training_logger.py`.

The system avoids heavy external tracking services (such as MLflow, Weights & Biases, or WandB servers) while ensuring 100% reproducible, auditable, and structured experiment persistence on disk.

---

## 2. Telemetry Directory Hierarchy

Every forecasting experiment run creates a dedicated directory under:
```
experiments/forecasting/lstm_training/<run_id>/
│
├── metadata.json              # Hardware, environment, package versions, threads
├── experiment_config.yaml     # Serialized experiment configuration
├── training_history.json      # Step-by-step epoch metrics array
├── training_summary.json      # Final run status, best epoch, duration
├── errors.json                # Execution anomalies or exceptions (if any)
└── checkpoints/               # Model weights and training state
    ├── checkpoint_best.pt     # Best validation loss state
    └── checkpoint_last.pt     # Final epoch state
```

---

## 3. Telemetry Schemas

### 3.1 Environment Metadata (`metadata.json`)
```json
{
  "run_id": "official_lstm_training",
  "created_at": "2026-11-20T06:59:03.984024+00:00",
  "platform": "Windows-11-10.0.26200-SP0",
  "python_version": "3.13.3 (tags/v3.13.3:6144f0f, Apr 8 2025, 19:41:40) [MSC v.1943 64 bit (AMD64)]",
  "torch_version": "2.15.0+cpu",
  "numpy_version": "2.2.6",
  "cpu_count": 8,
  "torch_threads": 2,
  "cuda_available": false
}
```

### 3.2 Epoch Metric Telemetry (`training_history.json`)
Each epoch appends a record adhering to this schema:
```json
{
  "epoch": 30,
  "train_loss": 0.0002559798548821958,
  "val_loss": 0.0004697327579682072,
  "learning_rate": 0.001,
  "epoch_duration_seconds": 0.03901863098144531,
  "batch_count": 13,
  "gradient_norm": 0.0015099849202670157,
  "early_stopped": false,
  "timestamp": "2026-11-20T06:59:06.090333+00:00"
}
```

### 3.3 Experiment Summary (`training_summary.json`)
At the conclusion of training, the final summary is emitted:
```json
{
  "run_id": "official_lstm_training",
  "status": "COMPLETED",
  "total_epochs": 30,
  "best_epoch": 30,
  "best_val_loss": 0.0004697327579682072,
  "total_duration_seconds": 2.117845296859741,
  "termination_reason": "Completed all scheduled epochs",
  "completed_at": "2026-11-20T06:59:06.102650+00:00"
}
```

---

## 4. Checkpoint State Schema

Checkpoints are serialized using PyTorch's native dictionary format:
```python
{
    "epoch": 30,
    "val_loss": 0.0004697327579682072,
    "best_epoch": 30,
    "best_val_loss": 0.0004697327579682072,
    "is_best": True,
    "model_state_dict": OrderedDict([...]),
    "optimizer_state_dict": dict([...]),
    "scheduler_state_dict": dict([...]),
    "experiment_id": "official_lstm_training",
    "metadata": {
        "train_loss": 0.0002559798548821958,
        "val_loss": 0.0004697327579682072,
        "learning_rate": 0.001,
        "epoch_duration_seconds": 0.03901863098144531
    }
}
```

---

## 5. Version Control & Git Hygiene

- **Source Code, Documentation & Reports:** Tracked in Git under `src/forecasting/`, `tests/forecasting/`, `docs/`, and `reports/`.
- **Checkpoints and Weights (`*.pt`):** Strictly excluded by `.gitignore` (`experiments/*`, `*.pt`).
- **Telemetry Artifacts:** Stored in local experiment runs for reproducibility and audit reporting.

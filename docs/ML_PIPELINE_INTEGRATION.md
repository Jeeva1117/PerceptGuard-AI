# Machine Learning Pipeline Integration Specification
## PerceptGuard AI: Predictive Reliability Engineering

---

## 1. Scientific Foundations & Objective

PerceptGuard AI forecasts future perception reliability to anticipate object-detection failures before they manifest downstream in autonomous vehicle trajectory planners.

### Ground-Truth Formulation
Ground-truth reliability is defined as spatial overlap against true annotations:
$$R_{t+h} = \text{IoU}(B_{t+h}, G_{t+h}) \in [0.0, 1.0]$$

- $B_{t+h}$: Predicted bounding box at forward horizon $h$.
- $G_{t+h}$: Ground-truth annotated bounding box at forward horizon $h$.

> [!IMPORTANT]
> YOLO detection confidence $P(\text{class}|\text{box})$ measures classification certainty and is treated strictly as an input feature, NOT ground truth. True reliability is derived from spatial IoU overlap.

---

## 2. Model Checkpoints & Execution

1. **Detection Model:** YOLO11n (`models/yolo/yolo11n.pt`, $5.6\text{ MB}$).
   - Pretrained on COCO and autonomous driving classes.
   - Evaluated on CPU with confidence threshold $0.25$ and NMS IoU threshold $0.45$.
2. **Forecasting Model:** LSTMForecaster (`models/forecasting/lstm/model.pt`, $103.5\text{ KB}$).
   - Architecture: 2 LSTM layers, hidden dimension $64$, input dimension $27$, linear output head predicting 3 continuous values ($H1, H3, H5$).
   - Trained on BDD100K multi-horizon sequences with EarlyStopping.

---

## 3. Reliability Risk Mapping

Reliability risk score $S$ represents failure potential:
$$S_{t+h} = 1.0 - \hat{R}_{t+h} \in [0.0, 1.0]$$

| Risk Score ($S$) | Risk Category | Operational Implication |
| :---: | :---: | :--- |
| $[0.00, 0.25)$ | **LOW** | Perception nominal; high anticipated spatial overlap. |
| $[0.25, 0.50)$ | **MEDIUM** | Moderate uncertainty; sensor or motion perturbation suspected. |
| $[0.50, 0.75)$ | **HIGH** | Significant risk of spatial localization degradation. |
| $[0.75, 1.00]$ | **CRITICAL** | Severe detection failure imminent; fallback initiated. |
| Stale / Missing | **UNKNOWN** | Predictions uncalculated; never defaulted to Low. |

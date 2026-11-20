# Known Operational Limitations & Calibration Status
## PerceptGuard AI: Anticipating Detection Failures Before They Happen

---

## 1. Hardware & Execution Limitations

1. **CPU Execution Throughput:**
   - On an Intel Core i3 11th Gen processor without GPU acceleration, full end-to-end inference (YOLO11n detection, ByteTrack tracking, 27-dimensional feature extraction, PyTorch LSTM inference, and HUD frame annotation) averages $125\text{ ms}$ to $488\text{ ms}$ per frame ($2.05\text{ to } 8.0\text{ fps}$).
   - **Recommendation:** Full 30 fps real-time processing requires dedicated GPU hardware or lightweight INT8/ONNX model quantization.

2. **Host Docker CLI Absence:**
   - The development Windows 11 host machine does not have the Docker Engine CLI installed. Multi-stage container definitions (`Dockerfile.backend`, `Dockerfile.frontend`, `docker-compose.yml`) are verified statically and tested remotely on GitHub Actions Ubuntu runners.

3. **Single Video Processing Concurrency:**
   - Background jobs are executed via a single-worker thread pool (`ThreadPoolExecutor(max_workers=1)`) to avoid CPU core thrashing and protect the 8 GB system RAM boundary. Concurrent job submissions are automatically queued.

---

## 2. Machine Learning & Calibration Limitations

1. **Provisional Calibration Gate:**
   - The reliability risk score $S = 1.0 - R$ is calculated directly from multi-horizon predicted reliability scores $R \in [0.0, 1.0]$.
   - Because the initial training sequence utilized a single driving clip with zero-inflated failure distributions, risk classification boundaries (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) remain under **`PROVISIONAL CALIBRATION`** status.
   - **Operator Notice:** Risk categories reflect predictive trends and relative degradation severity rather than calibrated empirical failure probabilities.

2. **Tracking Ground-Truth Independence:**
   - Real-world videos without MOT17/BDD100K-MOT ground-truth annotations are evaluated for tracking continuity (persistent IDs, displacement statistics) rather than MOTA, MOTP, or HOTA tracking accuracy.

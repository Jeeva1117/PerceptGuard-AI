# Real-World Testing & Operational Scenario Validation
## PerceptGuard AI: Anticipating Detection Failures Before They Happen

---

## 1. Overview & Test Scope

Day 14 validated the complete end-to-end PerceptGuard AI perception reliability pipeline against realistic operational scenarios extracted from authentic autonomous driving sequences (**BDD100K driving dataset**, reference video `converted_026c7465-309f6d33.mp4`, 1280x720, 30.1 fps).

All verifications were executed under resource-constrained local hardware constraints:
- **Processor:** Intel Core i3 11th Gen (CPU-only, zero CUDA dependency).
- **System Memory:** 8 GB RAM.
- **Operating System:** Windows 11 AMD64.

---

## 2. Tested Environmental Conditions & Scenarios

| Scenario | Operational Condition | Video Resolution & Size | Frames | Processing Latency | Key Perception & Reliability Findings |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Scenario A** | Normal Daytime Driving | 720x1280 (657.8 KB) | 15 | $7.32\text{s}$ ($488.0\text{ ms/frame}$) | Detected 3 vehicles; maintained persistent ByteTrack ID; generated forecasts ($H1=0.5976, H3=0.5896, H5=0.4455$, Category: MEDIUM). Nominal baseline behavior. |
| **Scenario B** | Low Lighting / Night | 720x1280 (219.5 KB) | 15 | $2.43\text{s}$ ($161.9\text{ ms/frame}$) | 0 detections due to darkness; feature extractor measured high contrast drop; LSTM forecaster escalated $H1$ risk to $0.7455$ and $H5$ to $0.6871$ (**Category: HIGH**). Anticipated failure successfully. |
| **Scenario C** | Motion-Heavy Driving | 720x1280 (1,051.9 KB) | 15 | $1.97\text{s}$ ($131.5\text{ ms/frame}$) | High displacement across frames; velocity trends detected; $H3$ failure risk escalated to $0.6637$ (**Category: HIGH**). Triggered 16 early-warning alerts. |
| **Scenario D** | Degraded Visual Quality | 720x1280 (168.3 KB) | 15 | $1.88\text{s}$ ($125.3\text{ ms/frame}$) | Gaussian blur ($\sigma=11$) and contrast reduction; Laplacian variance plummeted; $H1$ risk scored $0.7062$ (**HIGH**) and $H5$ scored $0.6022$ (**HIGH**). |
| **Scenario E** | Edge Case: Single Frame | 720x1280 (53.2 KB) | 1 | $0.17\text{s}$ ($172.1\text{ ms}$) | Processed 1-frame edge case without crash; sequence padded deterministically; initial risk scored cleanly. |
| **Scenario F** | Edge Case: Corrupted Stream | Invalid Stream (0.03 KB) | 0 | $0.01\text{s}$ | Handled gracefully: raised explicit `ValueError` (`OpenCV could not open video stream`); job transitioned to `FAILED` with descriptive error. |

---

## 3. Computer Vision & Reliability Pipeline Invariants Verified

1. **Zero Fake Detections:** When visual visibility drops to near-zero (Scenario B), the detector outputs 0 bounding boxes rather than hallucinating detections.
2. **Physics-Consistent Kinematics:** Track coordinates remain strictly within image dimensions $[0, W] \times [0, H]$. Track IDs increment deterministically from 1.
3. **Multi-Horizon Risk Dynamics:** Risk score $S = 1.0 - R$ correctly scales with visual quality degradation and motion uncertainty. Severe blur and low lighting trigger HIGH-risk alerts across $H1, H3, H5$.
4. **Memory Stability:** Host process RSS memory peaked at $527.9\text{ MB}$, with zero unbounded memory growth or tensor leakage during repeated video runs.

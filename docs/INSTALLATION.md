# PerceptGuard AI — Installation & Setup Guide

## 1. System Requirements

### 1.1 Minimum Hardware Specification
- **Processor:** Intel Core i3 (11th Gen or equivalent dual-core CPU with AVX2 support)
- **Memory (RAM):** 8 GB RAM minimum (System uses $< 600\text{ MB}$ RSS during peak inference)
- **Disk Space:** 2 GB free disk space for models, datasets, and virtual environment
- **GPU:** **None required.** The entire pipeline is engineered for 100% pure CPU execution.

### 1.2 Operating System & Software
- **Operating System:** Windows 10/11 (64-bit), Ubuntu 20.04+, or macOS 12+
- **Python Version:** Python 3.10, 3.11, 3.12, or 3.13 (Verified on Python 3.13.3)
- **Package Manager:** `pip` (v23.0+)

---

## 2. Step-by-Step Installation

### Step 1: Clone the Repository
Open a terminal (PowerShell on Windows, Bash on Linux/macOS):
```bash
git clone https://github.com/Jeeva1117/PerceptGuard-AI.git
cd PerceptGuard-AI
```

### Step 2: Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

On Linux / macOS (Bash):
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Upgrade pip and install the production dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Model Checkpoints
Ensure that the neural network model weights are present in the `models/` directory:
- **YOLO11n Object Detector:** `models/yolo/yolo11n.pt` ($5.6\text{ MB}$)
- **PyTorch LSTM Forecaster:** `models/forecasting/lstm/model.pt` ($103.5\text{ KB}$)

To verify file existence:
```powershell
# Windows PowerShell
Test-Path models\yolo\yolo11n.pt
Test-Path models\forecasting\lstm\model.pt
```

### Step 5: Verify System Environment
Run the built-in system verification utility:
```bash
python main.py --info
python main.py --verify-env
```
Expected output:
```
Base Directory       : .../PerceptGuard-AI
Selected Device      : cpu
Base Detection Model : yolo11n.pt
Temporal History Len : 10 frames
Forecasting Horizons : [1, 3, 5] frames ahead
Status: Full Day 1–15 Pipeline Implemented & Validated.
```

---

## 3. Starting the Application

### 3.1 Start the FastAPI Backend Server
In your first terminal (with `venv` activated):
```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```
Verify the backend is live:
- Swagger Documentation: `http://127.0.0.1:8000/docs`
- Root Health Probe: `http://127.0.0.1:8000/health`

### 3.2 Start the Streamlit Operations Dashboard
In a second terminal (with `venv` activated):
```bash
streamlit run dashboard/app.py
```
The dashboard interface will automatically open in your default browser at `http://localhost:8501`.

> [!NOTE]
> If the FastAPI backend is not started, the Streamlit dashboard automatically activates its **In-Process Fallback Mode**. All features, video processing, and forecasting capabilities remain fully operational within the single process.

---

## 4. Troubleshooting & Common Questions

### Issue: `PermissionError` on Windows during Ultralytics Initialization
- **Cause:** Ultralytics checks for a global configuration file at `%APPDATA%\Ultralytics\settings.json`.
- **Solution:** Ensure your PowerShell session has permission to read and write to your local user AppData folder. Running PowerShell as a regular user (or granting directory permissions) resolves this immediately.

### Issue: Memory consumption during long video runs
- **Cause:** Processing high-resolution video (> 720p) with hundreds of frames on 8 GB machines.
- **Solution:** In the `🎥 Video Processing` dashboard tab, set **Max Frames to Process** to a bounded value (e.g. `30` or `50` frames). The pipeline processes at $\approx 18.5\text{ FPS}$ on CPU with memory strictly bounded under $530\text{ MB}$ RSS.

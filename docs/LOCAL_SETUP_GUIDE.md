# Local Setup & Operational Guide
## PerceptGuard AI: Predictive Perception Reliability System

---

## 1. Prerequisites

- **Python Version:** Python 3.11, 3.12, or 3.13.
- **Operating System:** Windows 11, macOS, or Linux (Ubuntu 22.04+).
- **Hardware Minimum:** 4 vCPUs, 8 GB RAM, 10 GB free disk space.
- **Acceleration:** Pure CPU execution; CUDA is optional and not required.

---

## 2. Environment Setup

1. **Activate Virtual Environment:**
   ```powershell
   .\venv\Scripts\activate
   ```
2. **Install Dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Configure Environment:**
   ```powershell
   Copy-Item .env.example .env
   ```

---

## 3. Running the Application

### Option A: Running Decoupled Services (Recommended)

1. **Start the FastAPI Backend Service:**
   ```powershell
   .\venv\Scripts\uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
   Verify backend health: `curl http://127.0.0.1:8000/health/ready`

2. **Start the Streamlit Frontend Console:**
   ```powershell
   .\venv\Scripts\streamlit run dashboard/app.py --server.port 8501
   ```
   Open your browser to: `http://localhost:8501`

### Option B: Unified In-Process Fallback Mode

If the FastAPI backend is not running, launching Streamlit directly will automatically initialize the in-process `DashboardService`, allowing full local evaluation without running a separate HTTP server process.

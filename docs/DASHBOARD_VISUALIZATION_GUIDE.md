# PerceptGuard AI — Dashboard Visualization Guide

## 1. Visual Styling Philosophy

The PerceptGuard AI dashboard adheres to an enterprise dark-mode aesthetic inspired by the **Catppuccin Mocha** palette. The interface is optimized for high-contrast legibility, reduced operator eye fatigue during continuous monitoring, and clear semantic distinction between nominal operations and urgent failure states.

### Core Color Palette
| Semantic Meaning | Hex Code | Purpose |
| :--- | :--- | :--- |
| **Base Background** | `#11111b` | Main canvas background |
| **Surface 0 (Card)**| `#181926` | Card, container, and accordion background |
| **Surface 1 (Border)**| `#313244` | Subtle borders, dividers, and card outlines |
| **Text (Primary)** | `#cdd6f4` | High-contrast headlines and primary metrics |
| **Text (Muted)** | `#a6adc8` | Secondary labels, descriptions, and captions |
| **Text (Dim)** | `#6c7086` | Minor annotations, timestamps, and parameters |

---

## 2. Operational Risk Color Hierarchy

The multi-horizon risk scoring engine categorizes anticipated failure probability into four operational severity tiers. These colors are strictly applied across metrics, cards, chart markers, and alert badges:

| Severity Level | Hex Code | Color Swatch | Mathematical Range | Visual Treatment |
| :--- | :--- | :--- | :--- | :--- |
| **LOW** | `#2a9d8f` | Teal / Cyan | $[0.00, 0.35)$ | Solid green border, dark green badge |
| **MEDIUM** | `#e9c46a` | Amber / Gold | $[0.35, 0.65)$ | Solid gold border, dark gold badge |
| **HIGH** | `#f4a261` | Coral / Orange | $[0.65, 0.85)$ | Solid orange border, orange badge |
| **CRITICAL** | `#e76f51` | Crimson / Red | $[0.85, 1.00]$ | Solid red border, flashing crimson badge |
| **UNKNOWN** | `#6c757d` | Slate Gray | `None` / `NaN` | Gray border, muted neutral badge |

---

## 3. Data Visualization Taxonomy

### 3.1 Multi-Horizon Risk Cards (`risk_cards.py`)
- **Card Structure:** 3-column layout showing H1 ($t+1$), H3 ($t+3$), and H5 ($t+5$).
- **Key Indicators:** Large formatted score (`0.1500`), severity pill badge, rolling linear trend indicator (`INCREASING` ↗, `DECREASING` ↘, `STABLE` →), rolling mean, and observation count.

### 3.2 Temporal Risk Trajectory Chart (`temporal_chart.py`)
- **Chart Type:** Multi-line temporal plot.
- **Data Series:**
  - H1 Risk: Green line (`#2a9d8f`)
  - H3 Risk: Gold line (`#e9c46a`)
  - H5 Risk: Coral line (`#e76f51`)
- **X-Axis:** Video frame index / temporal step ($t$).
- **Y-Axis:** Failure risk $S \in [0.0, 1.0]$.
- **Reference Lines:** Horizontal dotted indicators at $0.35$ (MEDIUM threshold), $0.65$ (HIGH threshold), and $0.85$ (CRITICAL threshold).

### 3.3 Visual Quality Analytics (`feature_analytics.py`)
- **Blur Variance:** Continuous line plot of Laplacian variance ($\sigma^2_{\text{Laplacian}}$). Sharp drops indicate motion blur or defocussing.
- **Brightness & Contrast:** Dual-line overlay comparing mean pixel intensity and RMS contrast across frames.
- **Kinematic Speed:** Line plot tracking mean object velocity across Kalman frames. Spikes highlight rapid camera vibration or dynamic maneuvers.

### 3.4 Frame-by-Frame Bounding Box Viewer (`tracking_view.py`)
- **Interactive Slider:** Allows scrubbing through each processed frame of the video.
- **Visual Display:** Side-by-side layout:
  - Left column: High-resolution annotated image with colored bounding boxes and track ID labels.
  - Right column: Structured inspection table showing detection confidence %, class name, and persistent Kalman tracker identity.

---

## 4. Responsive UI & Screen Layout Rules

1. **Wide Layout Mode:** Forced via `st.set_page_config(layout="wide")` ensuring optimal dashboard density on desktop monitors ($1920\times 1080$ and $1440\times 900$).
2. **Column Balancing:** Metric rows are organized in grids of 2, 3, or 4 columns with normalized widths.
3. **Empty States:** When no data exists, components invoke `render_empty_state()` featuring centered iconography, clear explanatory prose, and actionable guidance rather than collapsed or missing UI sections.

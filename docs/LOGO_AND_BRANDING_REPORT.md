# PerceptGuard-AI — Logo & Brand Identity Report

This report documents the creative concept, vector design specifications, asset generation, frontend integration, and GitHub branding for **PerceptGuard-AI**.

---

## 1. Brand Concept & Design Philosophy

**Project Name**: `PerceptGuard-AI`  
**Official Tagline**: `See the Road. Predict the Risk. Prevent the Collision.`

### Design Metaphor
The symbol is an original geometric fusion of five core engineering ideas:
1. **Protective Shield Geometry**: Outer faceted shield contour symbolizing vehicle safety, structural protection, and reliability engineering.
2. **Perspective Road Corridor**: Converging perspective roadway lines originating from the bottom of the shield, symbolizing continuous forward travel and lane geometry.
3. **Sleek Autonomous Vehicle Silhouette**: A modern, aerodynamic vehicle profile featuring a rooftop perception sensor pod (LiDAR) and a continuous cybernetic LED lightbar signature.
4. **Predictive Perception Field**: Expanding radar/optical cones with concentric dashed horizon arcs representing multi-horizon predictive forecasting ($H1$, $H3$, $H5$).
5. **Computer Vision Aperture**: A calibrated optical crosshair / reticle centered at the focal horizon, representing computer vision precision.

---

## 2. Generated Asset Deliverables

All brand assets were engineered as pure, scalable SVG vectors with resolution-independent geometry:

| Asset Path | Canvas Size | Description |
| :--- | :--- | :--- |
| [`frontend/public/brand/perceptguard-icon.svg`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/frontend/public/brand/perceptguard-icon.svg) | 512 x 512 px | High-resolution primary square brand icon, suitable for application headers, avatars, and app icons. |
| [`frontend/public/brand/perceptguard-logo.svg`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/frontend/public/brand/perceptguard-logo.svg) | 840 x 200 px | Full horizontal branding banner featuring the symbol, bold geometric wordmark (`PERCEPTGUARD`), glowing `AI` distinction badge, and official tagline. |
| [`frontend/public/favicon.svg`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/frontend/public/favicon.svg) | 64 x 64 px | Micro-scale browser favicon optimized for crisp 16x16 and 32x32 tab display across all modern browsers. |
| [`frontend/public/brand/favicon.svg`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/frontend/public/brand/favicon.svg) | 64 x 64 px | Redundant archive copy of the favicon in the brand assets directory. |

---

## 3. Frontend Integration

In strict accordance with the constraint to preserve the existing UI:
- **No Layout Changes**: Spacing, margins, colors, and responsive grid layouts remain untouched.
- **Navigation Rail**: In `frontend/src/components/Sidebar.tsx`, the generic Lucide shield was replaced with the authentic `perceptguard-icon.svg` within the exact existing `h-10 w-10` boundary.
- **Browser Favicon & Title**: In `frontend/index.html`, the generic template title was updated to `PerceptGuard-AI — Predictive Road Safety`, and linked to `/favicon.svg`.

---

## 4. GitHub Repository Branding

The root [`README.md`](file:///c:/Users/acer/Desktop/PerceptGuard-AI/README.md) was updated with a centered hero branding section:
- Crisp SVG vector banner (`frontend/public/brand/perceptguard-logo.svg`).
- Official tagline: *See the Road. Predict the Risk. Prevent the Collision.*
- Authentic technological badges (Python 3.13, PyTorch CPU, Ultralytics YOLO11n, ByteTrack, FastAPI, React 19).
- Clean architectural documentation and local execution guides.

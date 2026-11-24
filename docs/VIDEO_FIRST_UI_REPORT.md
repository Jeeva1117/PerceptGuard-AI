# PERCEPTGUARD-AI — VIDEO-FIRST UI ARCHITECTURE & INTERACTION REPORT

**Document ID:** PG-REP-004  
**Date:** 2026-11-24  
**Author:** Coordinated Frontend & UX Architecture Engineering Team  
**Status:** VALIDATED & DEPLOYED  

---

## 1. Executive Summary & Design Rationale

Prior to this engineering redesign, PerceptGuard-AI operated as a fragmented, page-oriented web dashboard. Navigating between the five primary modules caused the HTML5 `<video>` element to unmount, destroying playback buffers, resetting scrub positions to frame 0, and forcing users to re-upload or re-select video sources.

The **Video-First UI Redesign** transforms PerceptGuard-AI into a cohesive, automotive-grade continuous video perception platform:
- The uploaded road traffic video is permanently mounted as the central visual workspace.
- The user can scrub, play, pause, and inspect the continuous video stream while freely navigating between all five primary modules without interruption.
- A standardized **Master Top Status Banner** provides instantaneous ADAS safety status.
- An **Interactive Multi-Horizon Timeline** visualizes timestamped perception events and enables one-click seeking.
- **Bidirectional Global Track Synchronization** links data tables directly to the canvas overlay.

---

## 2. Architecture & Layout Specification

### 2.1 Two-Column Persistent Workspace Layout

In `frontend/src/App.tsx`, the layout structure was re-architected from individual full-page route views into a synchronized dual-column split:

```
+----------------------------------------------------------------------------------------------------+
|  HEADER: App Title | Master Status Banner (Driving / Collision / Risk) | Calibration | Engine Stat |
+----------------------------------------------------------------------------------------------------+
| SIDEBAR         | PERSISTENT VIDEO WORKSPACE (Col 7/12)       | DYNAMIC MODULE SUB-ROUTER (Col 5/12)|
|                 |                                             |                                     |
| 1. Cam & Weath  | +-----------------------------------------+ | [Module Route Content]              |
| 2. Obj Detect   | |                                         | |                                     |
| 3. Trajectory   | |   Persistent HTML5 Video + Canvas       | | Module 1: Weather Telemetry         |
| 4. Coll Risk    | |   Synchronized ADAS Overlays            | | Module 2: Track Lifecycle Table     |
| 5. Safe Alerts  | |                                         | | Module 3: Multi-Horizon Forecasts   |
|                 | +-----------------------------------------+ | Module 4: TTC & Kinematic Matrix    |
|                 | | Playback Controls (Play, Scrub, Speed)  | | Module 5: Stopping Distance Advice  |
|                 | +-----------------------------------------+ |                                     |
|                 | | Click-to-Seek Interactive Timeline      | |                                     |
|                 | +-----------------------------------------+ |                                     |
+----------------------------------------------------------------------------------------------------+
```

### 2.2 Persistent Video Workspace (`PersistentVideoWorkspace.tsx`)
- Mounts `<VideoCanvas />`, `<PlaybackControls />`, and `<InteractiveTimeline />`.
- Wrapped outside the React Router `<Routes>` component.
- Guarantees zero component unmounting or DOM destruction when route changes occur from `/camera-weather` to `/object-detection`, `/trajectory-prediction`, `/collision-risk`, or `/safety-alerts`.

### 2.3 Dynamic Embedded Mode (`isEmbedded` prop)
All five module pages (`CameraWeatherPage`, `ObjectDetectionPage`, `TrajectoryPredictionPage`, `CollisionRiskPage`, and `SafetyAlertsPage`) were refactored to accept an `isEmbedded: boolean` prop:
- When rendered within the persistent workspace (`isEmbedded={true}`), duplicate video players, redundant upload drawers, and duplicate play buttons are automatically hidden.
- The page renders dedicated, high-density telemetry, metric cards, analytical plots, and tabular records tailored to that specific domain.

---

## 3. Core Component Capabilities

### 3.1 Standardized Master Top Status Banner (`Header.tsx`)
Positioned centrally in the application header, this banner provides instant executive-level driving condition awareness across three standardized badges:

1. **Driving Status Badge:**
   - `NORMAL`: Emerald badge (`bg-emerald-950/80 text-emerald-300 border-emerald-500/50`)
   - `CAUTION`: Amber badge (`bg-amber-950/80 text-amber-300 border-amber-500/50`)
   - `HAZARD`: Pulsing Rose badge (`bg-rose-950/90 text-rose-300 border-rose-500/60 animate-pulse`)
   - `UNKNOWN`: Slate badge (`bg-slate-800 text-slate-400 border-slate-700`)

2. **Collision Status Badge:**
   - `NO COLLISION RISK DETECTED`: Emerald badge with shield icon.
   - `COLLISION RISK DETECTED`: Blinking Rose badge with exclamation triangle icon.

3. **Risk Level Badge:**
   - Displays `LOW`, `MODERATE`, `HIGH`, or `CRITICAL` risk classification based on the active frame's time-to-collision and safety recommendation state.

### 3.2 Interactive Safety Timeline (`InteractiveTimeline.tsx`)
The interactive timeline provides a visual summary of perception and safety events across the entire duration of the video:
- **Timeline Bar:** Proportional horizontal bar mapping from $0.0\text{ s}$ to total video duration.
- **Playhead Indicator:** Glowing cyan marker reflecting current video playback timestamp $t$.
- **Click-to-Seek:** Clicking anywhere on the timeline invokes `seekVideo(targetTimestamp)` across the global `JobContext`, instantaneously updating the video and overlay canvas.
- **Categorical Event Markers:**
  - Sky Blue Pin: Weather degradation or optical visibility drop ($\text{visibility} < 70\%$).
  - Amber/Orange Pin: Pedestrian roadway activity or crossing awareness.
  - Red Blinking Pin: Collision risk or critical trajectory conflict ($\text{TTC} \le 3.5\text{s}$).
  - Yellow Pin: ADAS speed reduction advisory or stop recommendation.

### 3.3 Bidirectional Global Track Synchronization
To bridge the gap between visual video rendering and data tables:
1. `JobContext.tsx` maintains `selectedTrackId: number | null` and `setSelectedTrackId`.
2. In **Object Detection** (Module 2), **Trajectory Prediction** (Module 3), and **Collision Risk** (Module 4), clicking any row in the track tables invokes `setSelectedTrackId(track.track_id)`.
3. In `VideoCanvas.tsx`, the canvas render loop checks `selectedTrackId`:
   - The selected track's bounding box is rendered with a pulsing cyan border, larger font size, and glowing halo.
   - Its predicted multi-horizon trajectory ribbon ($0.5\text{s}, 1.0\text{s}, 1.5\text{s}, 2.0\text{s}$) is highlighted with high-contrast colored nodes.
   - All other tracks are rendered with reduced opacity, instantly focusing driver or analyst attention on the active subject.

---

## 4. Strict Navigation Boundary Verification

The application strictly enforces the 5-module hierarchy:
1. **Module 1:** Camera & Weather Intelligence (`/camera-weather`)
2. **Module 2:** Object Detection & Tracking (`/object-detection`)
3. **Module 3:** Object & Trajectory Prediction (`/trajectory-prediction`)
4. **Module 4:** Collision & Road Risk Assessment (`/collision-risk`)
5. **Module 5:** Early Warning & Vehicle Stop Recommendation (`/safety-alerts`)

- **Zero 6th Tab:** There is no separate "Video Upload" tab, "Settings" tab, or duplicate dashboard.
- **Inline Upload Drawer:** Video upload is cleanly integrated within the workspace header, allowing new video uploads while preserving layout geometry.

---

## 5. Build & Verification Status

The frontend was compiled and verified using Vite:
```bash
npm --prefix frontend run build
```
- **Modules Transformed:** 2,552 modules
- **TypeScript Errors:** 0
- **Build Status:** SUCCESS (production bundle generated in `frontend/dist/`)

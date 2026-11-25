# PerceptGuard-AI — Production UI Redesign & Workspace Architecture

**Document Version:** 1.0 (Production Release)  
**System Identity:** PerceptGuard-AI — Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  
**Core Scope Rule:** Strictly Five Primary Modules. No additional primary sidebar navigation tabs. Supported input: Uploaded video only.

---

## 1. Executive Summary of UI Re-Architecture

The user interface of PerceptGuard-AI has been transformed from an overcrowded split-view layout into a clean, professional, video-first engineering analysis workspace inspired by modern automotive perception, cloud telemetry, and autonomous driving workstations.

### Key Layout Transformations:
1. **Compact Navigation Rail (Screen Allocation: ~5%, `w-16 lg:w-[68px]`)**:
   - The previous oversized 288px (`w-72`) sidebar with verbose descriptive cards has been completely replaced with a sleek, vertical navigation rail.
   - Houses the brand identity icon and the **strictly five primary modules**:
     1. `Camera & Weather` (`CloudSun`)
     2. `Object Detection` (`Crosshair`)
     3. `Trajectory Prediction` (`TrendingUp`)
     4. `Collision Risk` (`AlertTriangle`)
     5. `Safety Alerts` (`OctagonAlert`)
   - Includes distinct active accent bars (`bg-cyan-400`), accessible labels (`aria-label`), keyboard navigation, and instant floating tooltips on hover.
   - Zero 6th primary tab exists in the codebase.

2. **Large Video-First Central Workspace (Screen Allocation: ~70%, `flex-1 min-w-0`)**:
   - The central video player now occupies the predominant portion of the screen, scaling dynamically while maintaining a crisp 16:9 (`aspect-video`) aspect ratio.
   - Houses the persistent **Video Control & Action Bar** at the top (`START PROCESSING` / `RE-RUN PROCESSING` trigger, Mode selector, Rotation button, and video telemetry pill).
   - Video canvas supports synchronized live overlays: bounding boxes, ByteTrack persistent IDs, trajectory projections with uncertainty radii, and focused track reticle.
   - Houses the unified playback controls and interactive timeline scrubber with timestamped hazard markers directly underneath the canvas.

3. **Context-Aware Module Details Panel (Screen Allocation: ~25%, `w-full xl:w-[380px] 2xl:w-[420px]`)**:
   - The right column displays **only the currently selected module's detailed output**.
   - Selecting any module in the navigation rail updates the right panel immediately without reloading, resetting, or pausing the video.
   - Video timestamp, active job, and overlay states remain perfectly synchronized across module transitions.

4. **Preserved Production Top Bar**:
   - Streamlined top bar preserving target stream selection, real-time driving status (`NORMAL`, `CAUTION`, `HAZARD`), collision status (`NO COLLISION RISK`, `NEAR PASS`, `POTENTIAL COLLISION RISK`, `COLLISION OBSERVED`, `UNKNOWN`), risk level, backend online indicator, and quick metrics.
   - Preserves the **Concise Master Analysis Result Banner** upon completion, displaying video duration, individual counts for all 6 road-user classes (`car`, `motorcycle`, `bus`, `truck`, `bicycle`, `pedestrian`), final collision risk, and evidence-based reason.
   - Safeguarded to strictly withhold `COMPLETED` banners if a job is in-flight, failed, or cancelled.

---

## 2. Wireframe & Visual Hierarchy

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TOP BAR: Target Stream | Driving: NORMAL | Collision: NEAR PASS | Risk: LOW | Backend  │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ MASTER SUMMARY: Video Analysis Completed | Duration: 2.5s | Road Users: 4 | Risk: SAFE │
├──────┬───────────────────────────────────────────────────┬─────────────────────────────┤
│      │  VIDEO CONTROL BAR: stream.mp4 | 02:29 | Mode     │  MODULE DETAILS PANEL       │
│ NAV  ├───────────────────────────────────────────────────┤                             │
│ RAIL │                                                   │  Selected Module Only:      │
│      │                                                   │  - Camera & Weather (M1)    │
│  M1  │               LARGE VIDEO WORKSPACE               │  - Object Detection (M2)    │
│  M2  │                                                   │  - Trajectory Predict (M3)  │
│  M3  │            Original + Synchronized Overlays       │  - Collision Risk (M4)      │
│  M4  │                 (16:9 Aspect Ratio)               │  - Safety Alerts (M5)       │
│  M5  │                                                   │                             │
│      │                                                   │  (Scrollable, Clean Cards,  │
│      ├───────────────────────────────────────────────────┤   Evidence-Based Rationale) │
│ CPU  │  PLAYBACK SCRUBBER & INTERACTIVE TIMELINE         │                             │
└──────┴───────────────────────────────────────────────────┴─────────────────────────────┘
```

---

## 3. Module Details Panel Specifications

### Module 1: Camera & Weather Intelligence
- **Stream Telemetry**: Actual video duration, resolution, frame rate (FPS), and container frames analyzed.
- **Optical Visibility Scoring**: Actual 0-100% visibility score, discrete category (`CLEAR`, `RAIN`, `FOG`, `LOW VISIBILITY`, `UNKNOWN`), and camera quality index.
- **Physical Optical Diagnostics**: Laplacian blur variance, mean brightness (/255), Michelson contrast, and sun-glare ratio.
- **Perception Evidence**: Transparent degradation reasons or confirmation of nominal optical clarity.
- **Data Integrity**: Unsupported placeholders (fake weather probabilities or unvalidated sensors) display `UNKNOWN` or `NOT AVAILABLE`.

### Module 2: Object Detection & Tracking
- **Road-User Distribution**: Dedicated counter chips for all 6 supported classes:
  - `Car`, `Motorcycle`, `Bus`, `Truck`, `Bicycle`, `Pedestrian`.
- **Active Detections (Current Timestamp)**: Real-time list of road users visible on the current video frame with track ID, confidence %, and speed. Includes a **Focus Track** button that highlights the agent in the central video player.
- **Track Registry Log**: Chronological list of all unique tracks observed across the video with clickable **Seek** actions jumping playback to their first-seen timestamp.

### Module 3: Trajectory Prediction
- **Lookahead Horizon Selector**: Discrete toggle for `H1` (+1.0s), `H3` (+3.0s), and `H5` (+5.0s) horizons.
- **Focused Track Kinematics**: Displays velocity ($v_x, v_y$), speed in km/h, movement direction (e.g. `Forward (North)`), and track history maturity.
- **Physics-Based Projection**: Reports predicted $(x, y)$ pixel coordinates and dynamic uncertainty radii. Tracks with $< 3$ frames history truthfully display `INSUFFICIENT DATA`.

### Module 4: Collision Risk Assessment
- **5-State Decision Taxonomy**: Strictly produces one of five mutually exclusive decisions:
  `NO COLLISION RISK`, `NEAR PASS`, `POTENTIAL COLLISION RISK`, `COLLISION OBSERVED`, `UNKNOWN`.
- **Evidence-Based Rationale**: Concise text explaining the physical geometry of the road corridor.
- **Spatio-Temporal Telemetry**: Validated Time-To-Collision (TTC) when closing, closing velocity in px/s or relative km/h, and lateral corridor clearance in meters and pixels.
- **Trajectory Conflict Points**: Lists synchronous intersecting pairs with delta time synchronicity ($\Delta t \le 0.5\text{s}$) and spatial conflict coordinates.

### Module 5: Safety Alerts & Advisories
- **Decision & Advisory Layout (Task 9 Standard)**:
  - Final Risk State: e.g. `NEAR PASS`
  - Relevant Timestamp: e.g. `2.50s`
  - Relevant Object: e.g. `None` or `#1 Motorcycle`
  - Evidence-Based Reason: e.g. `"Road user passes nearby without a credible path conflict."`
- **Compact 1-Line Filter Bar**: Quick filter pills (`All Events`, `Hazards`, `Advisories`, `Weather`) and search bar, replacing the bloated multi-dimensional filter panel.
- **Relevant Events Timeline**: Chronological event records with clickable `Seek to Event` controls.
- **Compact Research Disclaimer**: Readable, non-intrusive safety boundary callout at the footer.

---

## 4. Root-Cause Alert Deduplication Resolution (Task 12)

- **Audit Finding**: In `src/video_processing/video_pipeline_runner.py`, `all_alerts.extend(frame_alerts)` was executed on every single frame across all evaluated horizons. This caused persistent alerts to be counted repeatedly on every frame (e.g., 2 alerts $\times$ 98 frames = 196-198 alerts).
- **Engineering Fix**: Implemented alert instance deduplication by `alert_id` in `video_pipeline_runner.py`. The resulting `total_alerts` now reflects actual distinct alert events throughout the video rather than raw frame iteration counts.

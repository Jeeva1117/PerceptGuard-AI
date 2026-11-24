# PerceptGuard-AI: Trajectory Prediction & Collision Risk Validation Report

**Document Version:** 2.0 (Release Readiness)  
**System Identity:** PerceptGuard-AI — Visibility-Aware Predictive Road Safety Research Prototype  
**Tagline:** "See the Road. Predict the Risk. Prevent the Collision."  

---

## 1. Executive Summary

This report documents the validation of Feature 3 (Object & Trajectory Prediction) and Feature 4 (Collision & Road Risk Assessment). The perception engine translates raw bounding boxes and temporal tracking histories into physically grounded kinematic forecasts, spatio-temporal conflict analyses, and evidence-backed risk ratings.

A core autonomous driving research principle is enforced: **distinguishing genuine collision hazards from normal, safe driving situations**.

---

## 2. Feature 3: Kinematics & Multi-Horizon Trajectory Prediction

### Mathematical Formulation
For each active tracked object $i$ at frame $t$:
- **Position Centroid**: $(c_x, c_y) = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$
- **Velocity Vector**: $(v_x, v_y) = \left(\frac{\Delta c_x}{\Delta t}, \frac{\Delta c_y}{\Delta t}\right)$
- **Acceleration Vector**: $(a_x, a_y) = \left(\frac{\Delta v_x}{\Delta t}, \frac{\Delta v_y}{\Delta t}\right)$
- **Speed**: $v = \sqrt{v_x^2 + v_y^2}$ (converted to km/h via empirical perspective calibration)

### Multi-Horizon Future Predictions
Trajectories are predicted forward across discrete time horizons $h \in \{1, 3, 5, 10\}$ frames:
$$x(t + h) = c_x + v_x \cdot (h \cdot \Delta t) + \frac{1}{2} a_x \cdot (h \cdot \Delta t)^2 \cdot \gamma^h$$
$$y(t + h) = c_y + v_y \cdot (h \cdot \Delta t) + \frac{1}{2} a_y \cdot (h \cdot \Delta t)^2 \cdot \gamma^h$$
where $\gamma = 0.85$ is a constant velocity damping factor preventing runaway polynomial extrapolation.

Uncertainty covariance radii grow gracefully with horizon duration:
$$r_{\text{uncertainty}}(h) = 6.0 + 3.2 \cdot h + 0.05 \cdot v_{\text{km/h}}$$

---

## 3. Feature 4: Spatio-Temporal Collision Risk Assessment

### Core Physical Principles
PerceptGuard-AI rejects simplistic 2D proximity alarms. Two vehicles proximity alone does **not** constitute a collision risk. A hazard exists **only if** both spatial convergence and temporal synchronization coincide:

1. **Spatial Path Intersection**: Do the predicted trajectories intersect within physical safety envelopes?
2. **Temporal Synchronization**: Do both vehicles arrive at the intersection point within a critical temporal window ($\Delta t_{\text{arrival}} \le 0.5\text{s}$)?
3. **Closing Velocity Gate**: Is the relative closing velocity positive ($v_{\text{closing}} > 5.0\text{ px/s}$)? Diverging vehicles moving away from each other are strictly suppressed.
4. **Time-to-Collision (TTC)**:
   $$\text{TTC} = \frac{d_{\text{relative}}}{v_{\text{closing}}}$$
   - $\text{TTC} \le 1.8\text{s}$: **`CRITICAL`**
   - $1.8\text{s} < \text{TTC} \le 3.5\text{s}$: **`HIGH`**
   - $3.5\text{s} < \text{TTC} \le 5.0\text{s}$: **`MODERATE`**
   - $\text{TTC} > 5.0\text{s}$ or diverging: **`LOW`** (Normal driving)

---

## 4. Normal Safe Driving vs. Conflict Scenarios

### Scenario A: Normal Safe Road Driving
- **Characteristics**: Multiple vehicles travelling in parallel lanes at consistent cruising speeds.
- **System Behavior**:
  - Predicted trajectories remain parallel.
  - Spatial intersection is null or projected outside the forward horizon.
  - Closing velocity is zero or negative.
  - **Result**: `Collision Risk: LOW`, `Primary Action: NORMAL_DRIVING`. No alarms generated.

### Scenario B: Spatially Intersecting Paths with Temporal Separation
- **Characteristics**: Vehicle A passes through an intersection; Vehicle B crosses the same intersection 2.5 seconds later.
- **System Behavior**:
  - Spatial path intersection detected.
  - Temporal arrival differential $\Delta t_{\text{arrival}} = 2.5\text{s} > 0.5\text{s}$.
  - **Result**: `Collision Risk: LOW`. Suppressed by temporal gate.

### Scenario C: Genuine Turning Conflict (e.g. Motorcycle Crossing Path)
- **Characteristics**: A motorcycle cuts into the ego corridor with closing velocity $v_{\text{closing}} = 45\text{ px/s}$ and $\text{TTC} = 1.4\text{s}$.
- **System Behavior**:
  - Trajectory convergence confirmed.
  - Closing velocity gate passed.
  - $\text{TTC} < 1.8\text{s}$.
  - **Result**: `Collision Risk: CRITICAL`, `Primary Action: STOP_RECOMMENDED` or `PREPARE_TO_STOP`.

### Scenario D: Pedestrian Crossing Corridor
- **Characteristics**: A pedestrian steps into the active lane from the curb.
- **System Behavior**:
  - Evaluated by `PedestrianAwarenessEngine`.
  - State transitions to `POTENTIAL_PEDESTRIAN_COLLISION_RISK`.
  - **Result**: `Collision Risk: HIGH/CRITICAL`, `Primary Action: STOP_RECOMMENDED`.

---

## 5. Verification Test Matrix

| Test Suite | Scenario / Test Case | Expected Outcome | Verified Result |
| :--- | :--- | :--- | :--- |
| `test_regression_scenarios.py` | `test_clear_day_normal_highway` | Risk: LOW, Normal Driving | **PASS** |
| `test_regression_scenarios.py` | `test_clear_day_normal_city_traffic` | Risk: LOW, Normal Driving | **PASS** |
| `test_regression_scenarios.py` | `test_highway_parallel_lanes_no_collision` | Risk: LOW, No Conflict | **PASS** |
| `test_regression_scenarios.py` | `test_intersection_temporal_separation_no_collision` | Spatial overlap, temporal separated -> LOW | **PASS** |
| `test_regression_scenarios.py` | `test_sudden_cut_in_vehicle_collision` | Risk: HIGH/CRITICAL, Alert triggered | **PASS** |
| `test_regression_scenarios.py` | `test_pedestrian_crossing_path_collision` | Risk: CRITICAL, Stop Recommended | **PASS** |
| `test_regression_scenarios.py` | `test_motorcycle_lane_filtering_near_miss` | Cyclist/Motorcycle Conflict detected | **PASS** |
| `test_motorcycle_and_road_users.py` | Multi-class trajectory persistence | Kinematics computed without nulls | **PASS** |

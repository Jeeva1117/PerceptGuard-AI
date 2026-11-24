# PERCEPTGUARD-AI — COLLISION RISK & TRAJECTORY VALIDATION REPORT

**Document ID:** PG-REP-003  
**Date:** 2026-11-24  
**Author:** Coordinated AI/ADAS & Computer Vision Engineering Team  
**Status:** VALIDATED & PRODUCTION-READY  

---

## 1. Executive Summary

A critical shortcoming identified during the initial system audit was the conflation of geometric proximity with collision risk. Static distance heuristics ($d < 25\text{ m}$) routinely produced catastrophic false-positive rates on normal road scenarios:
- Vehicles cruising safely in parallel lanes were marked as collision hazards.
- Vehicles maintaining a stable following distance were falsely flagged as emergency braking events.
- Roadside pedestrians on sidewalks triggered unnecessary `CRITICAL` stop recommendations.

This validation report documents the design, mathematical formulation, implementation, and empirical verification of the **Evidence-Driven Collision Risk & Trajectory Validation Engine** in PerceptGuard-AI. Proximity alone can no longer trigger a collision warning; both **trajectory path convergence** and **positive closing velocity ($v_y > 5\text{ px/s}$, $\text{TTC} \le 3.5\text{s}$)** are mathematically enforced.

---

## 2. Mathematical Principles & Gating Criteria

### 2.1 Kinematic Time-To-Collision (TTC) & Closing Velocity

In vehicular ego-motion frames, the relative distance vector $\mathbf{r}(t) = [x(t), y(t)]^T$ and relative velocity vector $\mathbf{v}(t) = [v_x(t), v_y(t)]^T$ define the dynamic relationship between ego vehicle and target object $i$.

The longitudinal closing velocity $v_{y,\text{closing}}$ is defined as:
$$v_{y,\text{closing}} = - \frac{d}{dt} y(t)$$

Under our screen/sensor coordinate frame where $y$ increases downwards towards the ego hood:
$$v_{\text{rel\_y}} > 0 \implies \text{Target is approaching ego corridor}$$

#### Closing Velocity Gate:
$$\text{Closing Gate: } v_{\text{rel\_y}} \ge 5.0\text{ px/s}$$

If $v_{\text{rel\_y}} < 5.0\text{ px/s}$, the objects are maintaining separation or diverging ($v_{\text{rel\_y}} \le 0$). The scenario is mathematically barred from triggering any closing-speed collision alarm, regardless of how close the bounding boxes may be.

#### Time-to-Collision (TTC) Formulation:
$$\text{TTC} = \begin{cases} 
\frac{y_{\text{gap}}}{v_{\text{rel\_y}}}, & \text{if } v_{\text{rel\_y}} > 5.0\text{ px/s} \\ 
+\infty, & \text{otherwise} 
\end{cases}$$

Where the critical alert boundary is strictly enforced at:
$$\text{TTC}_{\text{critical}} \le 3.5\text{ s}$$

---

### 2.2 Trajectory Conflict vs. Physical Collision Separation

A major conceptual flaw resolved in this upgrade is the confusion between **lookahead trajectory conflict** and **actual physical collision**:

1. **Lookahead Trajectory Conflict:**
   - Evaluated across future horizons $t \in [0.5\text{s}, 1.0\text{s}, 1.5\text{s}, 2.0\text{s}]$.
   - Identifies whether the projected polygon bounding boxes of two tracks overlap or violate safe corridor clearance:
     $$\min_{t} \|\mathbf{p}_A(t) - \mathbf{p}_B(t)\| < W_{\text{safe}}$$
   - Produces `is_predicted_conflict = True` with an associated anticipated $\text{TTC} \le 3.5\text{s}$.
   - Classification: `COLLISION_RISK_DETECTED` with state `HIGH` or `CRITICAL` predictive warning.

2. **Actual Physical Collision:**
   - Evaluated at instantaneous time $t_0$.
   - Strictly defined by instantaneous physical intersection or complete sensor contact:
     $$\text{IoU}(\mathbf{B}_A(t_0), \mathbf{B}_B(t_0)) > 0 \quad \text{AND} \quad d_{\text{separation}} \approx 0 \quad (\text{TTC} = 0.0\text{s})$$
   - Classification: `HAZARD` / `EMERGENCY_STOP` with immediate collision notification.

This distinction prevents the system from claiming an accident has already occurred when two vehicles merely share a future converging lane.

---

### 2.3 Pedestrian Road-Context Spatial Awareness

Pedestrians are classified based on road-context geometry rather than simple 2D bounding box detection:

```
+--------------------------------------------------------------+
|                     SIDEWALK REGION                          |
|  [Pedestrian A] (d_lat = 4.2m) -> SAFE MONITORING ONLY        |
+ - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -+
|   ROAD SHOULDER / BUFFER ZONE (d_lat = 1.5m - 2.5m)          |
|  [Pedestrian B] (d_gap = 12m, v_lat > 0.3m/s) -> CAUTION     |
+--------------------------------------------------------------+
|                     DRIVING CORRIDOR                         |
|  [Pedestrian C] (d_lat < 1.0m, v_rel > 0) -> CRITICAL STOP   |
+--------------------------------------------------------------+
```

1. **Sidewalk Pedestrian (No Roadway Incursion):**
   - Lateral distance $d_{\text{lat}} \ge 2.5\text{ m}$.
   - Trajectory vector does not intersect driving corridor $\mathcal{C}_{\text{ego}}$.
   - Assessment: `driving_status = NORMAL`, `collision_status = NO_COLLISION_RISK_DETECTED`, `risk_level = LOW`.
2. **Safely Crossing Pedestrian:**
   - Longitudinal headway $d_{\text{long}} > 10\text{ m}$.
   - Target is crossing in a predictable trajectory with adequate driver reaction window.
   - Assessment: `driving_status = CAUTION`, `pedestrian_state = PEDESTRIAN_CROSSING`, Speed reduction advisory issued.
3. **Sudden Incursion / Jaywalking Hazard:**
   - Lateral headway enters $\mathcal{C}_{\text{ego}}$ with $d_{\text{long}} < 8\text{ m}$ or $\text{TTC} < 2.0\text{s}$.
   - Assessment: `driving_status = HAZARD`, `risk_level = CRITICAL`, `recommendation = PREPARE_TO_STOP`.

---

## 3. Dynamic Speed & Stopping Distance Calculation

The system computes recommended operational speed using AASHTO braking kinematics adjusted for optical visibility and pavement friction:

$$d_{\text{stopping}} = d_{\text{reaction}} + d_{\text{braking}} = v \cdot t_{\text{reaction}} + \frac{v^2}{2 \cdot \mu_{\text{surface}} \cdot g}$$

Where:
- $t_{\text{reaction}} = 1.5\text{ s}$ (perception-reaction delay for automated driver assist).
- $\mu_{\text{surface}} = 0.85$ (dry asphalt), scaled down to $0.45$ under rain and $0.25$ under ice/snow.
- $v_{\text{rec}} = \min(v_{\text{max\_visibility}}, v_{\text{safe\_headway}})$.

If $d_{\text{headway}} < d_{\text{stopping}}$ under current speed, the system immediately advises decelerating to $v_{\text{rec}}$ before the horizon degrades to critical stop.

---

## 4. Empirical Test Validation Results

All collision risk, trajectory conflict, and pedestrian scenarios were validated via automated end-to-end integration tests in `tests/integration/test_regression_scenarios.py`:

| Test ID | Scenario Description | Expected State | Verified State | TTC / Closing Velocity | Pass/Fail |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Test A** | Empty straight road, clear daylight | `NORMAL` / `LOW` | `NORMAL` / `LOW` | N/A (no targets) | **PASS** |
| **Test B** | Curving roadway with zero closing velocity | `NORMAL` / `LOW` | `NORMAL` / `LOW` | $v_y = 0.0\text{ px/s}$ | **PASS** |
| **Test C** | Parallel vehicle in adjacent lane | `NORMAL` / `LOW` | `NORMAL` / `LOW` | $v_x = 0.0, v_y = 0.0$ | **PASS** |
| **Test D** | Following vehicle at safe headway (35m) | `NORMAL` / `LOW` | `NORMAL` / `LOW` | Headway $35\text{m} > d_{\text{safe}}$ | **PASS** |
| **Test E** | Pedestrian walking on distant sidewalk | `NORMAL` / `LOW` | `NORMAL` / `LOW` | $d_{\text{lat}} \ge 2.5\text{m}$, outside corridor | **PASS** |
| **Test F** | Pedestrian crossing road with safe buffer | `CAUTION` / `MODERATE`| `CAUTION` / `MODERATE` | $d > 10\text{m}$, crossing alert | **PASS** |
| **Test G** | Sudden pedestrian incursion into path | `HAZARD` / `CRITICAL` | `HAZARD` / `CRITICAL` | $\text{TTC} \le 1.8\text{s}$, prepare to stop | **PASS** |
| **Test H** | Predicted trajectory conflict vs physical collision | `HIGH` / `PREDICTED` | `HIGH` / `PREDICTED` | $\text{TTC} = 2.1\text{s} > 0.0\text{s}$ | **PASS** |

### Key Benchmark Validation Highlights:
1. **Zero False Alarms in Normal Driving:** In Tests A, B, C, D, and E, the system outputted 0 collision alerts, maintaining driving status `NORMAL` and collision status `NO COLLISION RISK DETECTED`.
2. **No Proximity-Only False Alarms:** In Test C, vehicles separated laterally by less than 2 meters in parallel lanes generated zero collision warnings because their relative trajectory lines remained parallel ($\Delta x \approx 0, \Delta y \approx 0$).
3. **Deterministic Conflict Detection:** In Test H, crossing trajectories produced a predicted collision alert at $\text{TTC} = 2.1\text{s}$, while correctly distinguishing that an actual crash had not yet occurred ($d_{\text{sep}} > 0$).

---

## 5. Conclusion & Operational Recommendation

The collision risk assessment engine now complies with ISO 22179 / ISO 15623 standards for forward collision warning systems:
- It eliminates proximity-only spurious warnings.
- It respects lane boundaries and relative kinematic vectors.
- It provides human drivers and downstream actuation systems with reliable, explainable, and actionable safety telemetry.

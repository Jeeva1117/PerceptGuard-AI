export type JobStatus = 'QUEUED' | 'PROCESSING' | 'COMPLETED' | 'FAILED' | 'CANCELLED';
export type AnalysisMode = 'FULL_BALANCED' | 'FULL_DETAILED' | 'QUICK_PREVIEW';

export interface TimelineTelemetry {
  source_video_duration_sec: number;
  source_fps: number;
  total_container_frames: number;
  analysis_mode: string;
  effective_sampling_fps: number;
  total_frames_analyzed: number;
  first_analyzed_timestamp_sec: number;
  last_analyzed_timestamp_sec: number;
  timeline_coverage_percentage: number;
  is_full_duration_coverage: boolean;
  analysis_duration_realtime_sec: number;
  processing_speed_fps: number;
}

export interface TimelineEvent {
  event_id: string;
  event_type: 'WEATHER_FAULT' | 'COLLISION_HAZARD' | 'SAFETY_ADVISORY' | 'NORMAL_DRIVING' | string;
  category: string;
  severity: 'CRITICAL' | 'HIGH' | 'MODERATE' | 'LOW' | 'INFO' | string;
  start_timestamp_sec: number;
  end_timestamp_sec: number;
  start_time: string;
  end_time: string;
  duration_sec: number;
  evidence_frame_index: number;
  source_frame_index?: number;
  title: string;
  description: string;
  heuristic_metrics?: Record<string, any>;
  involved_tracks?: number[];
  action_recommended?: string;
}

export interface VideoUploadResponse {
  video_id: string;
  filename: string;
  size_bytes: number;
  format: string;
  metadata: {
    filename: string;
    width: number;
    height: number;
    fps: number;
    frame_count: number;
    duration_seconds: number;
    size_bytes: number;
  };
  stream_url: string;
  message: string;
}

export interface StartProcessingRequest {
  video_id: string;
  analysis_mode?: AnalysisMode;
  rotation?: number;
  sample_fps?: number;
  max_frames?: number | null;
}

export interface VideoMetadata {
  filename: string;
  width: number;
  height: number;
  fps: number;
  total_container_frames: number;
  duration_seconds: number;
  size_bytes?: number;
}

export interface JobStatusResponse {
  job_id: string;
  filename: string;
  status: JobStatus;
  stage: string;
  progress_pct?: number;
  progress_percentage?: number;
  message: string;
  error_message?: string | null;
  frames_processed: number;
  total_frames?: number;
  unique_tracks?: number;
  alerts_count?: number;
  total_detections?: number;
  created_at: string;
  completed_at?: string | null;
  result_summary?: Record<string, any>;
  video_path?: string | null;
  has_source_video?: boolean;
  rotation?: number;
  analysis_mode?: string;
  sample_fps?: number;
  timeline_telemetry?: TimelineTelemetry;
  event_timeline?: TimelineEvent[];
}

export interface Detection {
  bbox_xyxy: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
}

export type TrackLifecycleState = 'TENTATIVE' | 'CONFIRMED' | 'LOST' | 'REIDENTIFIED' | 'REMOVED';

export interface Track {
  track_id: number;
  bbox_xyxy: [number, number, number, number];
  confidence: number;
  class_id: number;
  class_name: string;
  start_frame?: number;
  age?: number;
  track_status?: string;
  track_lifecycle?: TrackLifecycleState;
}

export interface VisualQuality {
  blur_laplacian: number;
  brightness: number;
  contrast: number;
}

export interface RainEvidenceData {
  classification: string;
  severity: number;
  droplet_density: number;
  streak_variance: number;
  contrast_attenuation: number;
  details: string;
}

export interface FogEvidenceData {
  classification: string;
  severity: number;
  airlight_intensity: number;
  contrast_loss: number;
  depth_visibility_ratio: number;
  details: string;
}

export interface BlurEvidenceData {
  classification: string;
  severity: number;
  laplacian_variance: number;
  tenengrad_gradient_mean: number;
  edge_density: number;
  details: string;
}

export interface LightingEvidenceData {
  classification: string;
  severity: number;
  mean_luminance: number;
  underexposed_pixel_ratio: number;
  dynamic_range_spread: number;
  details: string;
}

export interface GlareEvidenceData {
  classification: string;
  severity: number;
  saturated_pixel_ratio: number;
  high_intensity_cluster_ratio: number;
  contrast_washout: number;
  details: string;
}

export interface CameraQualityMetricsData {
  sharpness: number;
  contrast: number;
  luminance: number;
  exposure: number;
  field_uniformity: number;
}

export interface PerceptionImpactData {
  expected_confidence_penalty: number;
  boundary_degradation_risk: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  small_object_risk: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  suggested_detection_sensitivity: 'NORMAL' | 'HEIGHTENED' | 'CONSERVATIVE';
  rationale: string;
  is_estimated: boolean;
}

export interface VisibilityForecastHorizon {
  horizon_frames: number;
  horizon_seconds: number;
  predicted_score: number;
  uncertainty_radius: number;
  confidence: number;
}

export interface VisibilityForecastData {
  degradation_trend:
    | 'Improving'
    | 'Stable'
    | 'Gradually degrading'
    | 'Rapidly degrading'
    | 'Uncertain'
    | 'INSUFFICIENT_DATA';
  trend_velocity: number;
  horizons: Record<string, VisibilityForecastHorizon>;
  confidence: number;
  uncertainty: number;
  forecast_method: string;
}

export interface WeatherIntelligence {
  visibility_score: number;
  visibility_level?: 'EXCELLENT' | 'GOOD' | 'DEGRADED' | 'CRITICAL';
  camera_quality_score?: number;
  primary_condition?: string;
  condition: string;
  is_degraded: boolean;
  condition_labels?: Record<string, string>;
  condition_severity?: Record<string, number>;
  rain?: RainEvidenceData;
  fog?: FogEvidenceData;
  blur?: BlurEvidenceData;
  lighting?: LightingEvidenceData;
  glare?: GlareEvidenceData;
  quality_metrics?: CameraQualityMetricsData;
  perception_impact?: PerceptionImpactData;
  forecast?: VisibilityForecastData;
  confidence?: number;
  uncertainty?: number;
  degradation_reasons?: string[];
  metrics?: {
    blur_laplacian: number;
    brightness: number;
    contrast: number;
    glare_ratio: number;
    edge_density: number;
    saturation?: number;
  };
  evidence?: string[];
}

export interface TrajectoryPoint {
  x: number;
  y: number;
  frame?: number;
  timestamp?: number;
}

export interface TrajectoryWaypointData {
  step_index: number;
  horizon_seconds: number;
  x: number;
  y: number;
  x_px: number;
  y_px: number;
  x_m?: number | null;
  y_m?: number | null;
  uncertainty_radius: number;
  uncertainty_radius_px: number;
  velocity_px_s: [number, number];
}

export interface TrajectoryHypothesisData {
  mode: 'PRIMARY_CONTINUATION' | 'LEFT_MANEUVER' | 'RIGHT_MANEUVER' | 'DECELERATING_STOP';
  mode_probability: number;
  intent_description: string;
  waypoints: TrajectoryWaypointData[];
}

export interface MultiAgentInteractionData {
  interaction_type: 'FREE_FLOW' | 'CAR_FOLLOWING' | 'CROSSING_PATH' | 'APPROACHING_OPPOSITE';
  nearest_agent_id: number | null;
  nearest_agent_distance_px: number | null;
  nearest_agent_distance_m: number | null;
  relative_speed_kmh: number | null;
  time_to_headway_seconds: number | null;
  interaction_notes: string;
}

export interface TrajectoryData {
  track_id: number;
  class_name: string;
  object_class?: string;
  confidence: number;
  track_lifecycle?: TrackLifecycleState;
  bbox: [number, number, number, number];
  center: [number, number];
  motion_state?: 'STATIONARY' | 'CRUISING' | 'ACCELERATING' | 'DECELERATING' | 'TURNING_LEFT' | 'TURNING_RIGHT' | 'UNSTABLE';
  speed_kmh: number;
  velocity: { vx: number; vy: number };
  acceleration: { ax: number; ay: number };
  heading_degrees?: number;
  track_age_frames?: number;
  history_length?: number;
  history: TrajectoryPoint[];
  observed_history?: TrajectoryPoint[];
  primary_trajectory?: TrajectoryWaypointData[];
  alternative_hypotheses?: TrajectoryHypothesisData[];
  predicted_horizons: Record<string, TrajectoryWaypointData>;
  horizons?: Record<string, TrajectoryWaypointData>;
  interaction?: MultiAgentInteractionData;
  uncertainty_factor?: number;
  prediction_status?: 'VALID' | 'INSUFFICIENT_HISTORY' | 'DEGRADED_PERCEPTION' | 'OCCLUDED_TRACK';
  coordinate_frame?: 'IMAGE_PIXEL' | 'APPROX_GROUND_PLANE_METRIC';
}

export interface ConflictPair {
  track_a: number;
  track_b: number;
  class_a: string;
  class_b: string;
  conflict_type?: string;
  time_to_conflict_seconds?: number;
  horizon_label?: string;
  horizon?: string;
  separation_pixels: number;
  separation_meters?: number | null;
  conflict_point: [number, number];
  conflict_point_m?: [number, number] | null;
  is_spatiotemporal_overlap?: boolean;
  combined_uncertainty_px?: number;
}

export type CollisionRiskState = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL' | 'UNKNOWN';

export interface CollisionRiskData {
  assessment_id?: string;
  timestamp_seconds?: number;
  risk_state: CollisionRiskState;
  risk_score?: number | null;
  primary_contributing_factors?: string[];
  objects_involved?: Array<{ track_id: number; object_class: string }>;
  min_ttc_seconds: number | null;
  closest_track_id: number | null;
  closest_distance_px?: number | null;
  closest_distance_m: number | null;
  relative_speed_kmh?: number | null;
  conflicting_pairs: ConflictPair[];
  uncertainty_level?: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  uncertainty_score?: number;
  camera_visibility_score?: number;
  detection_reliability?: number;
  explanation?: string;
  status?: string;
  perception_uncertainty_factor?: number;
  fused_risk_score?: number;
  rationale?: string;
}

export type RecommendationState =
  | 'NORMAL'
  | 'CAUTION'
  | 'REDUCE SPEED'
  | 'PREPARE TO STOP'
  | 'STOP RECOMMENDED'
  | 'UNKNOWN'
  | 'MAINTAIN_SPEED';

export type SafetyAction =
  | 'MAINTAIN_SPEED'
  | 'REDUCE_SPEED'
  | 'PREPARE_TO_STOP'
  | 'STOP_RECOMMENDATION'
  | string;

export type UrgencyLevel = 'INFO' | 'ADVISORY' | 'WARNING' | 'CRITICAL';

export interface InvolvedRoadAgent {
  track_id: number;
  object_class: string;
  distance_m?: number | null;
  speed_kmh?: number | null;
  relative_speed_kmh?: number | null;
}

export interface SupportingEvidenceDetail {
  trajectory_conflict: boolean;
  conflict_type?: string | null;
  min_ttc_seconds?: number | null;
  closest_distance_m?: number | null;
  prediction_horizon_seconds?: number | null;
  camera_visibility_score: number;
  camera_condition: string;
  detection_reliability: number;
  uncertainty_level: string;
  uncertainty_score: number;
}

export interface SafetyRecommendation {
  warning_id?: string;
  recommendation_state?: RecommendationState;
  primary_action: SafetyAction;
  priority?: UrgencyLevel;
  urgency_level: UrgencyLevel;
  risk_level?: string;
  stop_recommendation?: boolean;
  objects_involved?: InvolvedRoadAgent[];
  recommended_action?: string;
  primary_reason?: string;
  evidence_chain: string[];
  supporting_evidence?: SupportingEvidenceDetail;
  prediction_horizon?: string;
  min_ttc_seconds?: number | null;
  relative_speed_kmh?: number | null;
  camera_visibility_status?: string;
  detection_reliability?: number;
  uncertainty_level?: string;
  status?: string;
  disclaimer: string;
  timestamp_seconds: number;
}

export interface EarlyWarningAlert {
  alert_id?: string;
  rule_id: string;
  priority: 'P1' | 'P2' | 'P3';
  risk_level: string;
  horizon: number;
  message: string;
  timestamp?: string;
}

export type LaneState =
  | 'LANE DETECTED'
  | 'LANE CHANGE OBSERVED'
  | 'LANE DEPARTURE SUSPECTED'
  | 'LANE NOT VISIBLE'
  | 'UNKNOWN';

export type PedestrianRoadPosition = 'SIDEWALK' | 'CROSSWALK' | 'ROAD EDGE' | 'ROADWAY' | 'UNKNOWN';
export type PedestrianMovementIntent =
  | 'STATIONARY'
  | 'ALONG_SIDEWALK'
  | 'APPROACHING_ROAD'
  | 'CROSSING'
  | 'MOVING_AWAY'
  | 'UNKNOWN';

export type PedestrianAwarenessState =
  | 'PERSON DETECTED'
  | 'PEDESTRIAN APPROACHING ROAD'
  | 'PEDESTRIAN CROSSING DETECTED'
  | 'POTENTIAL PEDESTRIAN COLLISION RISK'
  | 'PEDESTRIAN TRACKING LOST'
  | 'NO PEDESTRIANS IN SCENE';

export interface RoadContextData {
  lane_state: LaneState;
  left_boundary_x: number;
  right_boundary_x: number;
  lane_center_px: number;
  lane_width_px: number;
  ego_lane_offset_px: number;
  road_surface_quality: string;
  crosswalk_detected: boolean;
  confidence: number;
}

export interface PedestrianDetail {
  track_id: number;
  bbox: [number, number, number, number];
  center: [number, number];
  confidence: number;
  road_position: PedestrianRoadPosition;
  movement_intent: PedestrianMovementIntent;
  state: PedestrianAwarenessState;
  risk_level: string;
  speed_kmh: number;
  distance_to_corridor_px: number;
  is_in_ego_path: boolean;
  ttc_seconds?: number | null;
  side: string;
  recommended_action: string;
  voice_prompt?: string | null;
  tracking_age_frames: number;
}

export interface PedestrianTelemetryData {
  frame_index: number;
  timestamp_seconds: number;
  pedestrians_count: number;
  primary_state: PedestrianAwarenessState;
  active_pedestrians: PedestrianDetail[];
  lost_pedestrian_ids: number[];
  highest_risk_level: string;
  recommended_action: string;
  primary_voice_alert?: string | null;
}

export interface FrameProcessingDetail {
  frame_index: number;
  timestamp_seconds: number;
  raw_frame_path: string;
  annotated_frame_path: string;
  detections_count: number;
  detections: Detection[];
  tracks_count: number;
  tracks: Track[];
  visual_quality: VisualQuality;
  predicted_reliability: Record<string, number>;
  risk_scores: Record<string, number>;
  risk_levels: Record<string, string>;
  alerts: EarlyWarningAlert[];
  latency_ms: number;
  source_frame_index?: number;
  weather_intelligence: WeatherIntelligence;
  trajectories: TrajectoryData[];
  collision_risk: CollisionRiskData;
  safety_recommendation: SafetyRecommendation;
  road_context?: RoadContextData;
  pedestrian_awareness?: PedestrianTelemetryData;
}

export interface VideoPipelineRunResult {
  job_id: string;
  video_filename: string;
  video_metadata: VideoMetadata;
  frames_processed: number;
  total_detections: number;
  unique_tracks: number;
  total_alerts: number;
  active_alerts: number;
  calibration_status: string;
  per_horizon_risk: Record<
    string,
    {
      horizon: number;
      latest_risk: number;
      mean_risk: number;
      max_risk: number;
      category: string;
    }
  >;
  frames: FrameProcessingDetail[];
  stage_durations_ms: Record<string, number>;
  total_duration_seconds: number;
  weather_summary: {
    total_frames_analyzed?: number;
    mean_visibility_score: number;
    min_visibility_score: number;
    predominant_condition?: string;
    condition_frequency?: Record<string, number>;
    dominant_conditions?: Record<string, number>;
    degraded_frames_count: number;
    degraded_frames_percentage?: number;
    trend_summary?: string;
    average_camera_quality?: number;
  };
  trajectory_summary?: {
    total_observed_trajectories: number;
    active_trajectories_count?: number;
    max_speed_kmh: number;
    mean_speed_kmh: number;
    total_track_points?: number;
    mean_ade_px?: number | null;
    mean_fde_px?: number | null;
    overall_ade_px?: number | null;
    overall_fde_px?: number | null;
    total_predicted_tracks?: number;
    motion_state_distribution?: Record<string, number>;
    coordinate_system?: string;
  };
  collision_summary?: {
    total_frames_assessed?: number;
    risk_state_distribution?: Record<string, number>;
    min_ttc_seconds: number | null;
    mean_risk_score?: number | null;
    critical_risk_frames: number;
    high_risk_frames: number;
    moderate_risk_frames?: number;
    low_risk_frames?: number;
    unknown_risk_frames?: number;
    total_conflicts_detected?: number;
    conflict_type_distribution?: Record<string, number>;
  };
  safety_summary: {
    total_recommendations?: number;
    recommendation_counts?: Record<string, number>;
    priority_counts?: Record<string, number>;
    most_frequent_recommendation?: string;
    critical_stop_count?: number;
    state_transitions?: Array<{
      from_state: string;
      to_state: string;
      timestamp_seconds: number;
      frame_index: number;
      trigger_reason: string;
    }>;
    action_counts: Record<string, number>;
    most_frequent_action: string;
    prototype_disclaimer: string;
  };
  road_context_summary?: Record<string, any>;
  pedestrian_summary?: Record<string, any>;
  timeline_telemetry?: TimelineTelemetry;
  event_timeline?: TimelineEvent[];
  video_path?: string | null;
  annotated_video_path?: string | null;
  completed_at: string;
}

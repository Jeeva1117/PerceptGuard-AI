# PerceptGuard-AI: Root-Cause Audit of Start Processing Button

**Document Version:** 1.0  
**Audit Target:** Start Processing Button Availability & Lifecycle State Machine  
**Category:** Frontend Architecture & State Reliability  
**Date:** 2026-11-25  

---

## 1. Executive Summary

During release-readiness usability audits, a critical frontend issue was identified:
> **"START PROCESSING appears sometimes, but disappears or is unavailable at other times."**

Observed symptoms from user telemetry and screenshots:
1. A video is uploaded and loaded into the continuous video player (`duration: 02:29`, `status: Ready for Upload`, `frames processed: 0`, `visibility: Unknown`).
2. The active stream shows `Active Stream: Stream Telemetry Loaded`.
3. The **Start Processing** button is completely missing from the screen.
4. Switching modules or reloading leaves the application in a state where analysis cannot be triggered without a fresh upload or modal workflow.

This audit details the exact code mechanisms responsible for this failure across `PersistentVideoWorkspace.tsx`, `VideoUploadSection.tsx`, `JobContext.tsx`, and the 5 embedded page views.

---

## 2. Root Cause Analysis

### Root Cause 1: Conditional Unmounting in `PersistentVideoWorkspace.tsx`
In `frontend/src/components/PersistentVideoWorkspace.tsx`, the component evaluated:
```tsx
const hasVideoLoaded = Boolean(activeJobId || results || stagedVideoSrc || uploadedVideoId);
```
Rendering logic:
```tsx
{!hasVideoLoaded ? (
  <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5 shadow-xl">
    <VideoUploadSection />
  </div>
) : (
  <div>
    <div className="flex items-center justify-between ...">
      <span>Active Stream: {activeJob?.filename || 'Stream Telemetry Loaded'}</span>
      <button onClick={() => setShowUploadDrawer((prev) => !prev)}>
        {showUploadDrawer ? 'Hide Upload Panel' : 'Upload Another Video'}
      </button>
    </div>
    {showUploadDrawer && (
      <div className="mb-4 bg-slate-900 ...">
        <VideoUploadSection />
      </div>
    )}
  </div>
)}
```
**Mechanism of Failure:**
1. The `<VideoUploadSection />` component contained the **only** instance of the "Start Processing" button in the main layout.
2. As soon as a video was staged or uploaded (`stagedVideoSrc` or `uploadedVideoId` became truthy), `hasVideoLoaded` evaluated to `true`.
3. Consequently, the component branched to the `else` block, rendering only the collapsed bar (`Active Stream: Stream Telemetry Loaded`) and hiding `<VideoUploadSection />`.
4. `showUploadDrawer` was initialized to `false`.
5. Therefore, the moment a video was loaded and ready to be processed, **the entire action area containing the "Start Processing" button was unmounted from the DOM**.
6. The operator was presented with the video player and a button labeled `"Upload Another Video"`, obscuring the ability to start processing the staged video.

---

### Root Cause 2: Fragile Local State in `VideoUploadSection.tsx`
`VideoUploadSection.tsx` did not rely on the global `JobContext` for its core state machine. Instead, it maintained redundant local state:
```tsx
const [selectedFile, setSelectedFile] = useState<File | null>(null);
const [uploadState, setUploadState] = useState<UploadStateMachine>('IDLE');
const [uploadedVideoId, setUploadedVideoId] = useState<string | null>(contextVideoId || null);
const [jobState, setJobState] = useState<JobExecutionState>('IDLE');
```
Button disabled condition (lines 655–659):
```tsx
disabled={
  uploadState === 'UPLOADING' ||
  uploadState === 'VALIDATING' ||
  (uploadState !== 'UPLOAD_COMPLETED' && !uploadedVideoId && !selectedFile)
}
```
**Mechanism of Failure:**
1. Whenever `<VideoUploadSection />` unmounted and remounted (e.g., when toggling the drawer, switching tabs, or re-rendering), all local `useState` variables were reset to their defaults (`selectedFile = null`, `uploadState = 'IDLE'`).
2. Even if `uploadedVideoId` had been issued by the backend, `uploadState` was `'IDLE'`.
3. If `uploadedVideoId` local state failed to synchronize with `contextVideoId`, the button rendered in a disabled state: `Start Processing (Upload Video First)`.

---

### Root Cause 3: Non-Rendering Across Sidebar Modules
In `frontend/src/App.tsx`, all five primary modules are mounted in the right column with `isEmbedded={true}`:
```tsx
<Route path="/camera-weather" element={<CameraWeatherPage isEmbedded />} />
<Route path="/object-detection" element={<ObjectDetectionPage isEmbedded />} />
<Route path="/trajectory-prediction" element={<TrajectoryPredictionPage isEmbedded />} />
<Route path="/collision-risk" element={<CollisionRiskPage isEmbedded />} />
<Route path="/safety-alerts" element={<SafetyAlertsPage isEmbedded />} />
```
In `CameraWeatherPage.tsx`, the upload section was gated:
```tsx
{!isEmbedded && <VideoUploadSection />}
```
And in the other 4 pages (`ObjectDetectionPage.tsx`, `TrajectoryPredictionPage.tsx`, `CollisionRiskPage.tsx`, `SafetyAlertsPage.tsx`), `VideoUploadSection` was neither imported nor rendered.

**Mechanism of Failure:**
- The persistent left workspace (`PersistentVideoWorkspace`) was the sole host of video ingestion and controls.
- Because `PersistentVideoWorkspace` hid `VideoUploadSection` whenever a video was staged, no module provided an accessible "Start Processing" action.

---

### Root Cause 4: Disconnection During Failure and Retry
When an analysis job failed (such as the earlier `ConflictType` error), `activeJob.status` was updated to `FAILED`.
1. `hasVideoLoaded` remained `true` (because `activeJobId` was set).
2. The upload drawer remained closed (`showUploadDrawer = false`).
3. The UI displayed a static failure banner without an accessible `Retry Processing` button.
4. The user had to guess how to reopen the upload drawer or refresh the browser.

---

## 3. Responsible Files

| File | Defect Summary |
| :--- | :--- |
| `frontend/src/components/PersistentVideoWorkspace.tsx` | Conditionally hides `<VideoUploadSection />` behind collapsed drawer when `hasVideoLoaded` is true. |
| `frontend/src/components/VideoUploadSection.tsx` | Duplicates video player, manages local state that resets on remount, and implements brittle disabled logic. |
| `frontend/src/context/JobContext.tsx` | Lacked a unified `VideoProcessingState` enum and centralized dispatch methods (`startProcessing`, `retryProcessing`). |
| `frontend/src/pages/CameraWeatherPage.tsx` | Suppresses `VideoUploadSection` when `isEmbedded` is true. |

---

## 4. Architectural Resolution Plan

1. **Centralize State in `JobContext.tsx`**:
   Implement a deterministic, single-source-of-truth state machine:
   ```
   NO_VIDEO -> UPLOADING -> VIDEO_READY -> PROCESSING -> COMPLETED
                                              │              │
                                              ▼              ▼
                                      PROCESSING_FAILED  CANCELLED
   ```
   Expose `videoProcessingState`, `startProcessing()`, `retryProcessing()`, `cancelProcessing()`, and `uploadFile()` directly from `JobContext`.

2. **Persistent Primary Action Area in `PersistentVideoWorkspace.tsx`**:
   - The action control toolbar must **NEVER** unmount.
   - It will render dynamically based on `videoProcessingState`:
     - `NO_VIDEO`: "Upload Driving Video" dropzone / file picker.
     - `UPLOADING`: Progress bar with percentage.
     - `VIDEO_READY`: Prominent, vibrant **"START PROCESSING"** button.
     - `PROCESSING`: Animated spinner with **"CANCEL PROCESSING"** option.
     - `COMPLETED`: Status badge with **"RE-RUN ANALYSIS"** or **"UPLOAD NEW VIDEO"**.
     - `PROCESSING_FAILED`: Clear error message with **"RETRY PROCESSING"** button.

3. **Eliminate Duplicate Video Previews**:
   Remove the redundant preview video tag inside `VideoUploadSection`. The single, continuous video element is hosted in `VideoCanvas` and remains mounted across all state transitions and module navigations.

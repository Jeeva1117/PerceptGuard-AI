import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { JobProvider } from './context/JobContext';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ErrorBoundary } from './components/ErrorBoundary';
import { PersistentVideoWorkspace } from './components/PersistentVideoWorkspace';
import { CameraWeatherPage } from './pages/CameraWeatherPage';
import { ObjectDetectionPage } from './pages/ObjectDetectionPage';
import { TrajectoryPredictionPage } from './pages/TrajectoryPredictionPage';
import { CollisionRiskPage } from './pages/CollisionRiskPage';
import { SafetyAlertsPage } from './pages/SafetyAlertsPage';

export const App: React.FC = () => {
  return (
    <ErrorBoundary moduleName="PerceptGuard AI System">
      <BrowserRouter>
        <JobProvider>
          <div className="flex h-screen w-screen overflow-hidden bg-slate-950 text-slate-100 font-sans">
            {/* Task 1: Compact Navigation Rail (~5% screen allocation) */}
            <Sidebar />

            {/* Right Main Application Area */}
            <div className="flex-1 flex flex-col h-screen min-w-0 overflow-hidden">
              {/* Task 3: Preserved Production Top Bar */}
              <Header />

              {/* Task 2: Large Video-First Production Workspace */}
              <main className="flex-1 overflow-hidden p-3 lg:p-4 bg-slate-950">
                <div className="h-full flex flex-col xl:flex-row gap-4 items-stretch overflow-hidden">
                  {/* Center Column: LARGE VIDEO WORKSPACE (~70% screen allocation) */}
                  <div className="flex-1 min-w-0 flex flex-col space-y-3 overflow-y-auto pr-1">
                    <PersistentVideoWorkspace />
                  </div>

                  {/* Right Column: CONTEXT-AWARE MODULE DETAILS PANEL (~25% screen allocation) */}
                  <aside className="w-full xl:w-[380px] 2xl:w-[420px] shrink-0 h-full overflow-y-auto pr-1">
                    <Routes>
                      <Route path="/" element={<Navigate to="/camera-weather" replace />} />
                      <Route
                        path="/camera-weather"
                        element={
                          <ErrorBoundary moduleName="Camera & Weather Intelligence">
                            <CameraWeatherPage isEmbedded />
                          </ErrorBoundary>
                        }
                      />
                      <Route
                        path="/object-detection"
                        element={
                          <ErrorBoundary moduleName="Object Detection & Tracking">
                            <ObjectDetectionPage isEmbedded />
                          </ErrorBoundary>
                        }
                      />
                      <Route
                        path="/object-tracking"
                        element={
                          <ErrorBoundary moduleName="Object Detection & Tracking">
                            <ObjectDetectionPage isEmbedded />
                          </ErrorBoundary>
                        }
                      />
                      <Route
                        path="/trajectory-prediction"
                        element={
                          <ErrorBoundary moduleName="Trajectory Prediction">
                            <TrajectoryPredictionPage isEmbedded />
                          </ErrorBoundary>
                        }
                      />
                      <Route
                        path="/collision-risk"
                        element={
                          <ErrorBoundary moduleName="Collision Risk Assessment">
                            <CollisionRiskPage isEmbedded />
                          </ErrorBoundary>
                        }
                      />
                      <Route
                        path="/safety-alerts"
                        element={
                          <ErrorBoundary moduleName="Safety Alerts & Recommendations">
                            <SafetyAlertsPage isEmbedded />
                          </ErrorBoundary>
                        }
                      />
                      <Route path="*" element={<Navigate to="/camera-weather" replace />} />
                    </Routes>
                  </aside>
                </div>
              </main>
            </div>
          </div>
        </JobProvider>
      </BrowserRouter>
    </ErrorBoundary>
  );
};

export default App;

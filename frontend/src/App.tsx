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
            {/* Exactly Five Primary Sidebar Features */}
            <Sidebar />

            {/* Right Main Application Area */}
            <div className="flex-1 flex flex-col h-screen min-w-0 overflow-hidden">
              <Header />

              {/* Master Video-First Continuous Workspace */}
              <main className="flex-1 overflow-y-auto p-4 lg:p-6 bg-slate-950/60">
                <div className="max-w-[1720px] mx-auto grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
                  {/* Left Column: Persistent Video, Scrubber & Interactive Timeline (Never Unmounts!) */}
                  <div className="xl:col-span-7 space-y-4">
                    <PersistentVideoWorkspace />
                  </div>

                  {/* Right Column: Active Module Specific Dynamic Telemetry */}
                  <div className="xl:col-span-5 space-y-4">
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
                  </div>
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

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { JobProvider } from './context/JobContext';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { ErrorBoundary } from './components/ErrorBoundary';
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

              {/* Scrollable Viewport */}
              <main className="flex-1 overflow-y-auto p-6 bg-slate-950/60">
                <div className="max-w-7xl mx-auto">
                  <Routes>
                    <Route path="/" element={<Navigate to="/camera-weather" replace />} />
                    <Route
                      path="/camera-weather"
                      element={
                        <ErrorBoundary moduleName="Camera & Weather Intelligence">
                          <CameraWeatherPage />
                        </ErrorBoundary>
                      }
                    />
                    <Route
                      path="/object-detection"
                      element={
                        <ErrorBoundary moduleName="Object Detection & Tracking">
                          <ObjectDetectionPage />
                        </ErrorBoundary>
                      }
                    />
                    <Route
                      path="/object-tracking"
                      element={
                        <ErrorBoundary moduleName="Object Detection & Tracking">
                          <ObjectDetectionPage />
                        </ErrorBoundary>
                      }
                    />
                    <Route
                      path="/trajectory-prediction"
                      element={
                        <ErrorBoundary moduleName="Trajectory Prediction">
                          <TrajectoryPredictionPage />
                        </ErrorBoundary>
                      }
                    />
                    <Route
                      path="/collision-risk"
                      element={
                        <ErrorBoundary moduleName="Collision Risk Assessment">
                          <CollisionRiskPage />
                        </ErrorBoundary>
                      }
                    />
                    <Route
                      path="/safety-alerts"
                      element={
                        <ErrorBoundary moduleName="Safety Alerts & Recommendations">
                          <SafetyAlertsPage />
                        </ErrorBoundary>
                      }
                    />
                    <Route path="*" element={<Navigate to="/camera-weather" replace />} />
                  </Routes>
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

import { HashRouter, Navigate, Route, Routes } from 'react-router-dom';
import React, { Suspense } from 'react';
import AppLayout from './layouts/AppLayout/AppLayout';
import Loading from './components/Loading/Loading';
import HomePage from './pages/HomePage';

const TellPage = React.lazy(() => import('./pages/TellPage'));
const ShiftPage = React.lazy(() => import('./pages/ShiftPage'));

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route
        path="/tell"
        element={
          <Suspense
            fallback={
              <AppLayout>
                <Loading />
              </AppLayout>
            }
          >
            <TellPage />
          </Suspense>
        }
      />
      <Route
        path="/shift"
        element={
          <Suspense
            fallback={
              <AppLayout>
                <Loading />
              </AppLayout>
            }
          >
            <ShiftPage />
          </Suspense>
        }
      />
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  );
}

function AppWithRouter() {
  return (
    <HashRouter>
      <App />
    </HashRouter>
  );
}

export default AppWithRouter;

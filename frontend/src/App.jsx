import { useState, useEffect } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import axios from 'axios'
import BASE from './lib/api'
import Login from './pages/Login'
import Signup from './pages/Signup'
import Dashboard from './pages/Dashboard'
import Chat from './pages/Chat'
import Onboarding from './pages/Onboarding'
import CheckIn from './pages/CheckIn'
import Async from './pages/Async'
import Profile from './pages/Profile'
import ProtectedRoute from './components/ProtectedRoute'
import PageTransition from './components/PageTransition'

function App() {
  const token = localStorage.getItem('token')
  const onboarded = localStorage.getItem('onboarded')
  const onboardingInProgress = localStorage.getItem('onboarding_progress')
  const [backendStatus, setBackendStatus] = useState('checking')

  useEffect(() => {
    axios.get(`${BASE}/`)
      .then(() => setBackendStatus('up'))
      .catch(() => setBackendStatus('down'))
  }, [])

  if (backendStatus === 'checking') {
    return (
      <div className="min-h-screen bg-parchment flex items-center justify-center">
        <div className="flex flex-col items-center gap-5 animate-fade-in">
          <div className="w-12 h-12 rounded-2xl bg-terra-dim border border-terra/20 flex items-center justify-center">
            <span className="font-display text-terra text-2xl italic">B</span>
          </div>
          <div className="flex gap-1.5">
            {[0, 150, 300].map(delay => (
              <span key={delay} className="w-1.5 h-1.5 bg-ink-ghost rounded-full animate-bounce" style={{ animationDelay: `${delay}ms` }} />
            ))}
          </div>
        </div>
      </div>
    )
  }

  if (backendStatus === 'down') {
    return (
      <div className="min-h-screen bg-parchment flex items-center justify-center px-6">
        <div className="text-center animate-fade-up max-w-sm">
          <div className="w-14 h-14 rounded-2xl bg-rose-dim border border-rose/20 flex items-center justify-center mx-auto mb-6">
            <span className="text-rose text-2xl">!</span>
          </div>
          <h2 className="font-display text-ink text-xl font-medium mb-2">Can't connect to BOND</h2>
          <p className="text-ink-dim text-sm mb-6 leading-relaxed">Make sure the backend server is running</p>
          <code className="block text-terra text-xs bg-parchment-warm border border-parchment-deeper px-4 py-3 rounded-xl mb-5">
            uvicorn main:app --reload
          </code>
          <button
            onClick={() => {
              setBackendStatus('checking')
              axios.get(`${BASE}/`).then(() => setBackendStatus('up')).catch(() => setBackendStatus('down'))
            }}
            className="text-ink-ghost text-sm hover:text-ink-muted transition font-display italic"
          >
            Try again →
          </button>
        </div>
      </div>
    )
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={
          !token ? <Navigate to="/login" /> :
          (!onboarded || onboardingInProgress) ? <Navigate to="/onboarding" /> :
          <Navigate to="/dashboard" />
        } />
        <Route path="/login" element={<PageTransition><Login /></PageTransition>} />
        <Route path="/signup" element={<PageTransition><Signup /></PageTransition>} />
        <Route path="/onboarding" element={
          <PageTransition><ProtectedRoute requireOnboarding={false}><Onboarding /></ProtectedRoute></PageTransition>
        } />
        <Route path="/checkin" element={
          <PageTransition><ProtectedRoute><CheckIn /></ProtectedRoute></PageTransition>
        } />
        <Route path="/dashboard" element={
          <PageTransition><ProtectedRoute><Dashboard /></ProtectedRoute></PageTransition>
        } />
        <Route path="/chat/:sessionType" element={
          <PageTransition><ProtectedRoute><Chat /></ProtectedRoute></PageTransition>
        } />
        <Route path="/async" element={
          <PageTransition><ProtectedRoute><Async /></ProtectedRoute></PageTransition>
        } />
        <Route path="/profile" element={
          <PageTransition><ProtectedRoute><Profile /></ProtectedRoute></PageTransition>
        } />
      </Routes>
    </BrowserRouter>
  )
}

export default App
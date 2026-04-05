import { Navigate } from 'react-router-dom'

export default function ProtectedRoute({ children, requireOnboarding = true }) {
  const token = localStorage.getItem('token')
  const onboarded = localStorage.getItem('onboarded')
  const onboardingInProgress = localStorage.getItem('onboarding_progress')

  if (!token) return <Navigate to="/login" replace />
  
  if (requireOnboarding && (!onboarded || onboardingInProgress)) {
    return <Navigate to="/onboarding" replace />
  }

  return children
}
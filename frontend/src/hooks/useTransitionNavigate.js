import { useNavigate } from 'react-router-dom'
import { useCallback } from 'react'

const EXIT_MS = 200

export default function useTransitionNavigate() {
  const navigate = useNavigate()

  const go = useCallback((to, options) => {
    const root = document.getElementById('root')
    if (!root) return navigate(to, options)

    root.classList.add('page-exiting')

    setTimeout(() => {
      root.classList.remove('page-exiting')
      navigate(to, options)
    }, EXIT_MS)
  }, [navigate])

  return go
}

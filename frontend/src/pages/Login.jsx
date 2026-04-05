import { useState } from 'react'
import useTransitionNavigate from '../hooks/useTransitionNavigate'
import { Link } from 'react-router-dom'
import axios from 'axios'
import BASE from '../lib/api'

export default function Login() {
  const [form, setForm] = useState({ email: '', password: '' })
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useTransitionNavigate()

  const handleSubmit = async () => {
    if (!form.email || !form.password) return
    setLoading(true)
    setError('')
    try {
      const res = await axios.post(`${BASE}/auth/login`, form)
      localStorage.setItem('token', res.data.token)
      localStorage.setItem('user_id', res.data.user_id)
      localStorage.setItem('name', res.data.name)
      localStorage.setItem('couples', JSON.stringify(res.data.couples))
      if (res.data.invite_code) localStorage.setItem('invite_code', res.data.invite_code)
      if (res.data.couples.length > 0) {
        localStorage.setItem('couple_id', res.data.couples[0].id)
        localStorage.setItem('active_couple', JSON.stringify(res.data.couples[0]))
      }
      if (res.data.is_onboarded) localStorage.setItem('onboarded', 'true')
      navigate('/dashboard')
    } catch (e) {
      setError(e.response?.data?.detail || 'Couldn\'t sign in — check your details')
    }
    setLoading(false)
  }

  return (
    <div className="min-h-screen bg-parchment flex flex-col lg:flex-row">

      {/* Left panel — decorative */}
      <div className="hidden lg:flex lg:w-1/2 bg-ink-soft relative overflow-hidden items-center justify-center p-16">
        <div className="absolute inset-0 opacity-5"
          style={{ backgroundImage: 'radial-gradient(circle at 30% 50%, #c4714a 0%, transparent 60%), radial-gradient(circle at 70% 80%, #7a9e8a 0%, transparent 50%)' }} />
        <div className="relative z-10 max-w-sm text-center animate-fade-in">
          <div className="w-14 h-14 rounded-2xl bg-terra-glow border border-terra/20 flex items-center justify-center mx-auto mb-8">
            <span className="font-display text-terra text-2xl italic">B</span>
          </div>
          <h1 className="font-display text-parchment text-4xl font-medium leading-tight mb-4">
            A quiet space for<br />
            <em>what matters most</em>
          </h1>
          <p className="text-ink-faint text-sm leading-relaxed">
            BOND listens to both of you separately, then helps you understand each other.
          </p>
          <div className="mt-12 space-y-3">
            {[
              'Private conversations with AI',
              'Understands both sides',
              'Builds insight over time',
            ].map((item, i) => (
              <div key={i} className="flex items-center gap-3 text-left">
                <div className="w-1.5 h-1.5 rounded-full bg-terra flex-shrink-0" />
                <p className="text-ink-ghost text-sm">{item}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right panel — form */}
      <div className="flex-1 flex items-center justify-center px-6 py-16 lg:py-0">
        <div className="w-full max-w-sm animate-fade-up">

          {/* Mobile logo */}
          <div className="lg:hidden text-center mb-10">
            <div className="w-12 h-12 rounded-2xl bg-terra-dim border border-terra/20 flex items-center justify-center mx-auto mb-4">
              <span className="font-display text-terra text-xl italic">B</span>
            </div>
            <h1 className="font-display text-ink text-3xl font-medium">BOND</h1>
          </div>

          <div className="mb-8">
            <h2 className="font-display text-ink text-2xl font-medium mb-1">Welcome back</h2>
            <p className="text-ink-dim text-sm">Good to have you here</p>
          </div>

          {error && (
            <div className="bg-rose-dim border border-rose/20 rounded-2xl px-4 py-3 mb-6 animate-slide-up">
              <p className="text-rose text-sm">{error}</p>
            </div>
          )}

          <div className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-ink-dim mb-1.5 tracking-wide uppercase">Email</label>
              <input
                type="text"
                placeholder="you@example.com"
                value={form.email}
                onChange={e => setForm({ ...form, email: e.target.value })}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                autoComplete="email"
                className="w-full bg-white/60 border border-parchment-darker rounded-2xl px-4 py-3.5 text-ink placeholder-ink-ghost focus:outline-none focus:border-terra/40 text-sm transition"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-ink-dim mb-1.5 tracking-wide uppercase">Password</label>
              <input
                type="password"
                placeholder="••••••••"
                value={form.password}
                onChange={e => setForm({ ...form, password: e.target.value })}
                onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                autoComplete="current-password"
                className="w-full bg-white/60 border border-parchment-darker rounded-2xl px-4 py-3.5 text-ink placeholder-ink-ghost focus:outline-none focus:border-terra/40 text-sm transition"
              />
            </div>

            <button
              onClick={handleSubmit}
              disabled={loading || !form.email || !form.password}
              className="w-full bg-ink hover:bg-ink-soft disabled:opacity-40 disabled:cursor-not-allowed text-parchment font-medium py-3.5 rounded-2xl transition text-sm tracking-wide mt-2 shadow-soft"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-parchment/20 border-t-parchment rounded-full animate-spin" />
                  Signing in...
                </span>
              ) : 'Sign in'}
            </button>
          </div>

          <div className="flex items-center gap-4 my-6">
            <div className="flex-1 divider" />
            <span className="text-ink-ghost text-xs">or</span>
            <div className="flex-1 divider" />
          </div>

          <p className="text-center text-ink-dim text-sm">
            New here?{' '}
            <Link to="/signup" className="text-terra hover:text-terra-light font-medium transition">
              Create an account
            </Link>
          </p>

          <p className="text-center text-ink-ghost text-xs mt-8 leading-relaxed">
            Your conversations are private and never shared
          </p>
        </div>
      </div>
    </div>
  )
}
import { useState, useEffect } from 'react'
import useTransitionNavigate from '../hooks/useTransitionNavigate'
import axios from 'axios'
import BASE from '../lib/api'

function ArrowLeftIcon() {
  return <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><polyline points="15 18 9 12 15 6"/></svg>
}

const PROFILE_FIELDS = [
  { key: 'communication_style', label: 'How you communicate' },
  { key: 'conflict_style', label: 'During conflict' },
  { key: 'support_style', label: 'What helps you most' },
  { key: 'love_language', label: 'How you feel loved' },
  { key: 'hope', label: 'What you hope for' },
]

const PATTERN_FIELDS = [
  { key: 'recurring_themes', label: 'Themes that come up' },
  { key: 'triggers', label: 'What tends to trigger you' },
  { key: 'what_helps', label: 'What helps you settle' },
  { key: 'avoidance_patterns', label: 'What you tend to avoid' },
]

function SkeletonLine({ w = 'w-full' }) {
  return <div className={`h-4 ${w} rounded skeleton`} />
}

export default function Profile() {
  const navigate = useTransitionNavigate()
  const token = localStorage.getItem('token')
  const name = localStorage.getItem('name')
  const [profile, setProfile] = useState(null)
  const [patterns, setPatterns] = useState(null)
  const [loading, setLoading] = useState(true)
  const [sessionCount, setSessionCount] = useState(0)
  const [inviteCode, setInviteCode] = useState(null)
  const [copied, setCopied] = useState(false)

  const activeCouple = (() => {
    try { return JSON.parse(localStorage.getItem('active_couple')) } catch { return null }
  })()

  const copyCode = () => {
    if (!inviteCode) return
    navigator.clipboard.writeText(inviteCode)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  useEffect(() => {
    const fetchAll = async () => {
      try {
        const [profileRes, sessionsRes, refreshRes] = await Promise.all([
          axios.get(`${BASE}/profile/me?token=${token}`),
          activeCouple
            ? axios.get(`${BASE}/sessions/${activeCouple.id}?token=${token}`)
            : Promise.resolve({ data: { sessions: [] } }),
          axios.post(`${BASE}/auth/login-refresh`, { token }),
        ])
        const p = profileRes.data.profile
        setProfile(p)
        if (p?.patterns) setPatterns(p.patterns)
        setSessionCount((sessionsRes.data.sessions || []).filter(s => !s.is_active).length)
        // Get invite code from the user's own couple (the one without a partner, or first one)
        const couples = refreshRes.data.couples || []
        const ownCouple = couples.find(c => !c.partner) || couples[0]
        if (ownCouple?.invite_code) setInviteCode(ownCouple.invite_code)
      } catch {}
      setLoading(false)
    }
    fetchAll()
  }, [])

  const profileFields = profile
    ? PROFILE_FIELDS.map(f => ({ ...f, value: profile[f.key] })).filter(f => f.value)
    : []

  const patternFields = patterns
    ? PATTERN_FIELDS.map(f => ({ ...f, value: patterns[f.key] })).filter(f => f.value && (!Array.isArray(f.value) || f.value.length > 0))
    : []

  return (
    <div className="min-h-screen bg-parchment">
      <div className="max-w-2xl mx-auto px-5 lg:px-8 py-8 lg:border-x lg:border-parchment-deeper min-h-screen">

        {/* Header */}
        <div className="flex items-center gap-3 mb-10">
          <button onClick={() => navigate('/dashboard')}
            className="w-9 h-9 rounded-xl bg-parchment-warm border border-parchment-deeper flex items-center justify-center text-ink-muted hover:text-ink transition">
            <ArrowLeftIcon />
          </button>
          <h1 className="font-display text-ink text-xl font-medium">Your profile</h1>
        </div>

        {/* Identity */}
        <div className="mb-10 flex items-end gap-4 pb-8 border-b border-parchment-deeper">
          <div className="w-16 h-16 rounded-2xl bg-parchment-deep border border-parchment-deeper flex items-center justify-center flex-shrink-0">
            <span className="font-display text-ink-muted text-3xl font-medium">
              {name?.charAt(0)?.toUpperCase()}
            </span>
          </div>
          <div>
            <h2 className="font-display text-ink text-3xl font-medium">{name}</h2>
            <div className="flex items-center gap-3 mt-1">
              <p className="text-ink-ghost text-sm">BOND member</p>
              {sessionCount > 0 && (
                <>
                  <span className="text-parchment-deeper">·</span>
                  <p className="text-ink-ghost text-sm font-display italic">{sessionCount} session{sessionCount !== 1 ? 's' : ''}</p>
                </>
              )}
            </div>
          </div>
        </div>

        {/* Invite code */}
        {inviteCode && (
          <div className="mb-10">
            <p className="text-xs font-medium text-ink-ghost tracking-widest uppercase mb-3">Your invite code</p>
            <div className="bg-white/60 rounded-2xl px-5 py-4 border border-parchment-darker flex items-center justify-between gap-4">
              <div>
                <p className="font-mono text-ink text-lg font-medium tracking-widest">{inviteCode}</p>
                <p className="text-ink-ghost text-xs mt-0.5">Share this with anyone you want to connect with</p>
              </div>
              <button onClick={copyCode}
                className="flex-shrink-0 text-xs text-ink-muted hover:text-ink border border-parchment-deeper hover:border-parchment-darker rounded-xl px-3 py-2 transition bg-parchment-warm">
                {copied ? '✓ Copied' : 'Copy'}
              </button>
            </div>
          </div>
        )}

        {/* What BOND knows */}
        <div className="mb-10">
          <p className="text-xs font-medium text-ink-ghost tracking-widest uppercase mb-5">What BOND knows about you</p>

          {loading ? (
            <div className="space-y-4">
              {[1, 2, 3].map(i => (
                <div key={i} className="bg-white/60 rounded-2xl p-4 border border-parchment-darker space-y-2">
                  <SkeletonLine w="w-1/4" />
                  <SkeletonLine w="w-3/4" />
                </div>
              ))}
            </div>
          ) : profileFields.length > 0 ? (
            <div className="space-y-3">
              {profileFields.map(field => (
                <div key={field.key} className="bg-white/60 rounded-2xl px-5 py-4 border border-parchment-darker">
                  <p className="text-ink-ghost text-2xs font-medium uppercase tracking-widest mb-1">{field.label}</p>
                  <p className="text-ink text-sm leading-relaxed">{field.value}</p>
                </div>
              ))}
            </div>
          ) : (
            <div className="bg-parchment-warm rounded-2xl p-6 border border-parchment-deeper text-center">
              <p className="font-display text-ink-dim text-base italic mb-1">No profile yet</p>
              <p className="text-ink-ghost text-sm">Complete onboarding to build your profile</p>
            </div>
          )}
        </div>

        {/* Behavioral intelligence */}
        {!loading && patternFields.length > 0 && (
          <div className="mb-10">
            <p className="text-xs font-medium text-ink-ghost tracking-widest uppercase mb-5">What BOND has learned from your sessions</p>
            <div className="bg-white/60 rounded-2xl border border-parchment-darker overflow-hidden">
              {patternFields.map((field, i) => {
                const items = Array.isArray(field.value) ? field.value : [field.value]
                const isLast = i === patternFields.length - 1
                return (
                  <div key={field.key} className={`px-5 py-4 ${!isLast ? 'border-b border-parchment-deeper' : ''}`}>
                    <p className="text-ink-ghost text-2xs font-medium uppercase tracking-widest mb-2">{field.label}</p>
                    <div className="flex flex-wrap gap-1.5">
                      {items.map((item, j) => (
                        <span key={j} className="text-xs bg-parchment-warm border border-parchment-deeper rounded-xl px-2.5 py-1 text-ink-muted">
                          {item}
                        </span>
                      ))}
                    </div>
                  </div>
                )
              })}
              {patterns?.watch_for && (
                <div className="px-5 py-4 border-t border-terra/20 bg-terra-dim">
                  <p className="text-terra text-2xs font-medium uppercase tracking-widest mb-1">Worth watching</p>
                  <p className="text-ink text-sm leading-relaxed font-display italic">{patterns.watch_for}</p>
                </div>
              )}
              {patterns?.self_awareness_trajectory && (
                <div className="px-5 py-4 border-t border-parchment-deeper">
                  <p className="text-ink-ghost text-2xs font-medium uppercase tracking-widest mb-1">Self-awareness</p>
                  <p className="text-ink text-sm leading-relaxed">{patterns.self_awareness_trajectory}</p>
                </div>
              )}
            </div>
            <p className="text-ink-ghost text-2xs mt-3 px-1">Updated after each session · Only visible to you</p>
          </div>
        )}

        {/* No sessions nudge */}
        {!loading && patternFields.length === 0 && sessionCount === 0 && profileFields.length > 0 && (
          <div className="mb-10 bg-parchment-warm rounded-2xl p-5 border border-parchment-deeper">
            <p className="font-display text-ink text-base font-medium mb-1">Behavioral intelligence</p>
            <p className="text-ink-ghost text-sm leading-relaxed">
              After your first session, BOND will start recognizing your patterns, triggers, and what helps you most.
            </p>
          </div>
        )}

        {/* Update */}
        <button
          onClick={() => { localStorage.removeItem('onboarded'); navigate('/onboarding') }}
          className="w-full border border-parchment-deeper hover:border-terra/30 text-ink-ghost hover:text-ink-muted py-4 rounded-2xl text-sm transition font-display italic"
        >
          Update your profile
        </button>

        <p className="text-center text-ink-ghost text-xs mt-5">
          Your profile helps BOND support you better
        </p>
      </div>
    </div>
  )
}
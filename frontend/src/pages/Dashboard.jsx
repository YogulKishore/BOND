import { useState, useEffect } from 'react'
import useTransitionNavigate from '../hooks/useTransitionNavigate'
import axios from 'axios'
import BASE from '../lib/api'

function timeOfDay() {
  const h = new Date().getHours()
  if (h < 5) return 'Late night'
  if (h < 12) return 'Good morning'
  if (h < 17) return 'Good afternoon'
  return 'Good evening'
}

function relativeTime(iso) {
  const diff = Math.floor((Date.now() - new Date(iso)) / 1000)
  if (diff < 60) return 'just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  if (diff < 604800) return `${Math.floor(diff / 86400)}d ago`
  return new Date(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

const PHASE_META = {
  listening:    { label: 'Listening',    dot: 'bg-sage' },
  understanding:{ label: 'Understanding',dot: 'bg-clay' },
  bridging:     { label: 'Bridging',     dot: 'bg-terra' },
  resolution:   { label: 'Resolution',   dot: 'bg-terra phase-dot-active' },
  integration:  { label: 'Integration',  dot: 'bg-terra' },
}

function CopyIcon() {
  return (
    <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
      <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
    </svg>
  )
}
function CheckIcon() {
  return (
    <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
      <polyline points="20 6 9 17 4 12"/>
    </svg>
  )
}
function TrashIcon() {
  return (
    <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
    </svg>
  )
}
function PlusIcon() {
  return (
    <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/>
    </svg>
  )
}
function LogOutIcon() {
  return (
    <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4"/><polyline points="16 17 21 12 16 7"/><line x1="21" y1="12" x2="9" y2="12"/>
    </svg>
  )
}
function SettingsIcon() {
  return (
    <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <circle cx="12" cy="12" r="3"/>
      <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z"/>
    </svg>
  )
}
function ChevronRightIcon() {
  return (
    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <polyline points="9 18 15 12 9 6"/>
    </svg>
  )
}
function UsersIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8}>
      <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/>
      <path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
    </svg>
  )
}

export default function Dashboard() {
  const navigate = useTransitionNavigate()
  const name = localStorage.getItem('name')
  const token = localStorage.getItem('token')
  const userId = localStorage.getItem('user_id')

  const [couples, setCouples] = useState(() => {
    try { return JSON.parse(localStorage.getItem('couples')) || [] } catch { return [] }
  })
  const [activeCouple, setActiveCouple] = useState(() => {
    try { return JSON.parse(localStorage.getItem('active_couple')) || null } catch { return null }
  })
  const [recentSessions, setRecentSessions] = useState([])
  const [hasUnreadAsync, setHasUnreadAsync] = useState(false)
  const [partnerSession, setPartnerSession] = useState(null) // active shared session started by partner
  const [sessionsLoading, setSessionsLoading] = useState(false)

  const [joinCode, setJoinCode] = useState('')
  const [copied, setCopied] = useState(null)
  const [joining, setJoining] = useState(false)
  const [joinError, setJoinError] = useState('')
  const [showJoin, setShowJoin] = useState(false)
  const [showNewCouple, setShowNewCouple] = useState(false)
  const [newLabel, setNewLabel] = useState('')
  const [creating, setCreating] = useState(false)
  const [mobilePanel, setMobilePanel] = useState('home') // home | sessions
  const [showSignOutConfirm, setShowSignOutConfirm] = useState(false)

  const fetchSessions = async (coupleId) => {
    if (!coupleId) return
    setSessionsLoading(true)
    try {
      const res = await axios.get(`${BASE}/sessions/${coupleId}?token=${token}`)
      setRecentSessions(res.data.sessions || [])
    } catch (e) {}
    setSessionsLoading(false)
  }

  useEffect(() => {
    if (couples.length > 0 && !activeCouple) {
      setActiveCouple(couples[0])
      localStorage.setItem('active_couple', JSON.stringify(couples[0]))
    }
  }, [couples])

  useEffect(() => {
    const refresh = async () => {
      try {
        const res = await axios.post(`${BASE}/auth/login-refresh`, { token })
        if (res.data.couples) {
          setCouples(res.data.couples)
          localStorage.setItem('couples', JSON.stringify(res.data.couples))
          if (activeCouple) {
            const updated = res.data.couples.find(c => c.id === activeCouple.id)
            if (updated) { setActiveCouple(updated); localStorage.setItem('active_couple', JSON.stringify(updated)) }
          }
        }
      } catch (e) {}
      if (activeCouple) {
        fetchSessions(activeCouple.id)
        try {
          const res = await axios.get(`${BASE}/async/messages/${activeCouple.id}?token=${token}`)
          const msgs = res.data.messages || []
          const readTs = JSON.parse(localStorage.getItem('async_read') || '{}')
          const lastRead = readTs[activeCouple.id]
          setHasUnreadAsync(msgs.some(m => m.sender_id !== userId && (!lastRead || new Date(m.created_at) > new Date(lastRead))))
        } catch (e) {}
      }
    }
    refresh()
    window.addEventListener('focus', refresh)
    return () => window.removeEventListener('focus', refresh)
  }, [activeCouple?.id])

  useEffect(() => { if (activeCouple) fetchSessions(activeCouple.id) }, [activeCouple?.id])
  useEffect(() => { if (activeCouple) fetchSessions(activeCouple.id) }, [])

  // Poll for partner session every 10 seconds
  useEffect(() => {
    if (!activeCouple) return
    const interval = setInterval(() => fetchSessions(activeCouple.id), 10000)
    return () => clearInterval(interval)
  }, [activeCouple?.id])

  const copyCode = (code, id) => {
    navigator.clipboard.writeText(code)
    setCopied(id)
    setTimeout(() => setCopied(null), 2000)
  }

  const handleJoin = async () => {
    setJoining(true); setJoinError('')
    try {
      await axios.post(`${BASE}/auth/join`, { token, invite_code: joinCode.toUpperCase() })
      // Refresh from server to get clean state — removes old solo couple, shows partner name
      const refresh = await axios.post(`${BASE}/auth/login-refresh`, { token })
      if (refresh.data.couples) {
        setCouples(refresh.data.couples)
        localStorage.setItem('couples', JSON.stringify(refresh.data.couples))
        // Auto-select the newly connected couple
        const connected = refresh.data.couples.find(c => c.partner)
        if (connected) {
          setActiveCouple(connected)
          localStorage.setItem('active_couple', JSON.stringify(connected))
          localStorage.setItem('couple_id', connected.id)
        }
      }
      setJoinCode(''); setShowJoin(false)
    } catch (e) { setJoinError(e.response?.data?.detail || 'Invalid code') }
    setJoining(false)
  }

  const handleNewCouple = async () => {
    if (!newLabel.trim()) return
    setCreating(true)
    try {
      const res = await axios.post(`${BASE}/auth/new-couple`, { token, label: newLabel.trim() })
      const updated = [...couples, res.data]
      setCouples(updated); localStorage.setItem('couples', JSON.stringify(updated))
      setShowNewCouple(false); setNewLabel('')
    } catch (e) {}
    setCreating(false)
  }

  const handleDeleteCouple = async (coupleId, e) => {
    e.stopPropagation()
    if (!window.confirm('Remove this connection?')) return
    try {
      await axios.post(`${BASE}/auth/leave-couple`, { token, couple_id: coupleId })
      const updated = couples.filter(c => c.id !== coupleId)
      setCouples(updated); localStorage.setItem('couples', JSON.stringify(updated))
      if (activeCouple?.id === coupleId) {
        const next = updated[0] || null
        setActiveCouple(next); localStorage.setItem('active_couple', JSON.stringify(next || ''))
      }
      if (updated.length === 0) setShowNewCouple(true)
    } catch (e) {}
  }

  const selectCouple = (couple) => {
    const fresh = couples.find(c => c.id === couple.id) || couple
    setActiveCouple(fresh)
    localStorage.setItem('active_couple', JSON.stringify(fresh))
    localStorage.setItem('couple_id', fresh.id)
  }

  const startSharedSession = async () => {
    const fc = couples.find(c => c.id === activeCouple.id) || activeCouple
    let freshCouple = fc
    try {
      const res = await axios.post(`${BASE}/auth/login-refresh`, { token })
      if (res.data.couples) {
        const updated = res.data.couples.find(c => c.id === fc.id)
        if (updated) {
          freshCouple = updated
          setCouples(res.data.couples)
          setActiveCouple(updated)
          localStorage.setItem('couples', JSON.stringify(res.data.couples))
          localStorage.setItem('active_couple', JSON.stringify(updated))
        }
      }
    } catch (e) {}

    // Check for existing active session first
    const activeShared = recentSessions.find(s => s.session_type === 'shared' && s.is_active && s.initiated_by === userId)
    if (activeShared) {
      return navigate(`/chat/shared?couple_id=${freshCouple.id}&session_id=${activeShared.id}`)
    }

    // Create session now so session_id is in the URL
    try {
      const res = await axios.post(`${BASE}/sessions/create?couple_id=${freshCouple.id}&session_type=shared&token=${token}`)
      const sessionId = res.data.session_id
      navigate(`/chat/shared?couple_id=${freshCouple.id}&session_id=${sessionId}`)
    } catch (e) {
      navigate(`/chat/shared?couple_id=${freshCouple.id}`)
    }
  }

  const activeSessions = recentSessions.filter(s => s.is_active)
  const partnerStartedSession = activeSessions.find(s => {
    if (s.session_type !== 'shared') return false
    if (!s.initiated_by || s.initiated_by === userId) return false
    // Only show if started in the last 24 hours
    const ageMs = Date.now() - new Date(s.created_at).getTime()
    return ageMs < 24 * 60 * 60 * 1000
  })
  const pastSessions = recentSessions.filter(s => !s.is_active)

  // ── Sidebar content (desktop left, or full page sections on mobile) ────────
  const SidebarContent = () => (
    <div className="space-y-8">

      {/* Connections */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs font-medium text-ink-ghost tracking-widest uppercase">Connections</p>
          <button onClick={() => { setShowJoin(!showJoin); setShowNewCouple(false) }} className="text-ink-ghost hover:text-ink-dim transition">
            <PlusIcon />
          </button>
        </div>

        {showJoin && (
          <div className="mb-3 bg-white/60 rounded-2xl p-4 border border-parchment-darker animate-slide-up">
            <p className="text-xs text-ink-ghost mb-2">Enter your partner's invite code</p>
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="INVITE CODE"
                value={joinCode}
                onChange={e => setJoinCode(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === 'Enter' && handleJoin()}
                autoFocus
                autoComplete="off"
                className="flex-1 bg-parchment-warm border border-parchment-darker rounded-xl px-3 py-2 text-ink text-xs uppercase tracking-widest focus:outline-none focus:border-terra/40 transition"
              />
              <button onClick={handleJoin} disabled={joining || !joinCode}
                className="bg-ink text-parchment text-xs px-3 py-2 rounded-xl disabled:opacity-40 transition font-medium">
                {joining ? '...' : 'Join'}
              </button>
            </div>
            {joinError && <p className="text-rose text-xs mt-2">{joinError}</p>}
          </div>
        )}

        {showNewCouple && (
          <div className="mb-3 bg-white/60 rounded-2xl p-4 border border-parchment-darker animate-slide-up">
            <input
              type="text"
              placeholder="Name this connection"
              value={newLabel}
              onChange={e => setNewLabel(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleNewCouple()}
              autoFocus
              className="w-full bg-parchment-warm border border-parchment-darker rounded-xl px-3 py-2 text-ink text-sm focus:outline-none focus:border-terra/40 transition mb-2"
            />
            <div className="flex gap-2">
              <button onClick={handleNewCouple} disabled={creating || !newLabel.trim()}
                className="flex-1 bg-ink text-parchment text-xs py-2 rounded-xl disabled:opacity-40 transition font-medium">
                {creating ? 'Creating...' : 'Create'}
              </button>
              <button onClick={() => { setShowNewCouple(false); setNewLabel('') }}
                className="flex-1 bg-parchment-deep text-ink-dim text-xs py-2 rounded-xl transition">
                Cancel
              </button>
            </div>
          </div>
        )}

        {couples.length === 0 && !showNewCouple && (
          <button onClick={() => setShowNewCouple(true)}
            className="w-full text-left px-4 py-3.5 rounded-2xl border border-dashed border-parchment-deeper text-ink-ghost text-sm hover:border-terra/30 hover:text-ink-dim transition">
            + Create your first connection
          </button>
        )}

        <div className="space-y-1.5">
          {couples.map(couple => (
            <div key={couple.id}
              onClick={() => selectCouple(couple)}
              className={`group flex items-center gap-3 px-3 py-3 rounded-2xl cursor-pointer transition-all ${
                activeCouple?.id === couple.id
                  ? 'bg-ink text-parchment'
                  : 'hover:bg-white/60 text-ink'
              }`}
            >
              <div className={`w-8 h-8 rounded-xl flex items-center justify-center flex-shrink-0 text-xs font-display font-medium ${
                activeCouple?.id === couple.id ? 'bg-parchment/10 text-parchment' : 'bg-parchment-deep text-ink-muted'
              }`}>
                {couple.label.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className={`text-sm font-medium truncate ${activeCouple?.id === couple.id ? 'text-parchment' : 'text-ink'}`}>
                  {couple.label}
                </p>
                <p className={`text-xs truncate ${activeCouple?.id === couple.id ? 'text-parchment/50' : 'text-ink-ghost'}`}>
                  {couple.partner ? `with ${couple.partner}` : couple.invite_code}
                </p>
              </div>
              <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition">
                {!couple.partner && (
                  <button onClick={e => { e.stopPropagation(); copyCode(couple.invite_code, couple.id) }}
                    className={`w-6 h-6 rounded-lg flex items-center justify-center ${activeCouple?.id === couple.id ? 'text-parchment/60 hover:text-parchment' : 'text-ink-ghost hover:text-ink-muted'} transition`}>
                    {copied === couple.id ? <CheckIcon /> : <CopyIcon />}
                  </button>
                )}
                <button onClick={e => handleDeleteCouple(couple.id, e)}
                  className={`w-6 h-6 rounded-lg flex items-center justify-center ${activeCouple?.id === couple.id ? 'text-parchment/60 hover:text-rose' : 'text-ink-ghost hover:text-rose'} transition`}>
                  <TrashIcon />
                </button>
              </div>
            </div>
          ))}
        </div>

        <button onClick={() => setShowJoin(!showJoin)}
          className="mt-2 w-full text-center text-ink-ghost text-xs py-2 hover:text-ink-dim transition">
          Have a partner's invite code? Join →
        </button>
      </div>
    </div>
  )

  // ── Session actions ────────────────────────────────────────────────────────
  const SessionActions = () => (
    <div className="space-y-3">

      {/* Partner started a session — notification card */}
      {partnerStartedSession && (
        <div className="w-full flex items-center gap-4 bg-terra-dim border border-terra/20 rounded-2xl p-4 animate-fade-up">
          <div className="w-10 h-10 rounded-xl bg-terra/20 flex items-center justify-center flex-shrink-0">
            <span className="font-display text-terra text-sm italic">!</span>
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-ink text-sm font-medium">{activeCouple?.partner || 'Your partner'} started a session</p>
            <p className="text-ink-ghost text-xs mt-0.5">They're waiting — join to talk with BOND together</p>
          </div>
          <div className="flex gap-2 flex-shrink-0">
            <button
              onClick={() => navigate(`/chat/shared?couple_id=${activeCouple?.id}&session_id=${partnerStartedSession.id}`)}
              className="bg-terra text-parchment text-xs px-3 py-2 rounded-xl font-medium hover:opacity-90 transition"
            >
              Join
            </button>
            <button
              onClick={() => setRecentSessions(prev => prev.filter(s => s.id !== partnerStartedSession.id))}
              className="text-ink-ghost text-xs px-3 py-2 rounded-xl hover:text-ink-muted transition"
            >
              Later
            </button>
          </div>
        </div>
      )}

      {/* Individual */}
      <button
        onClick={() => navigate(`/chat/individual?couple_id=${activeCouple?.id || 'solo'}`)}
        className="w-full group flex items-center gap-4 bg-white/60 hover:bg-white/90 border border-parchment-darker hover:border-sage/30 rounded-2xl p-4 text-left transition-all"
      >
        <div className="w-10 h-10 rounded-xl bg-sage-dim flex items-center justify-center flex-shrink-0">
          <span className="font-display text-sage text-sm italic">I</span>
        </div>
        <div className="flex-1">
          <p className="text-ink text-sm font-medium">Individual session</p>
          <p className="text-ink-ghost text-xs mt-0.5">A private space just for you</p>
        </div>
        <span className="text-ink-ghost group-hover:text-ink-dim transition"><ChevronRightIcon /></span>
      </button>

      {activeCouple && (
        <>
          {activeCouple.partner ? (
            <>
              {/* Shared */}
              <button
                onClick={startSharedSession}
                className="w-full group flex items-center gap-4 bg-white/60 hover:bg-white/90 border border-parchment-darker hover:border-terra/30 rounded-2xl p-4 text-left transition-all"
              >
                <div className="w-10 h-10 rounded-xl bg-terra-dim flex items-center justify-center flex-shrink-0">
                  <span className="font-display text-terra text-sm italic">S</span>
                </div>
                <div className="flex-1">
                  <p className="text-ink text-sm font-medium">Shared session</p>
                  <p className="text-ink-ghost text-xs mt-0.5">Talk together with BOND · with {activeCouple.partner}</p>
                </div>
                <span className="text-ink-ghost group-hover:text-ink-dim transition"><ChevronRightIcon /></span>
              </button>

              {/* Async */}
              <button
                onClick={() => navigate(`/async?couple_id=${activeCouple.id}`)}
                className="w-full group flex items-center gap-4 bg-white/60 hover:bg-white/90 border border-parchment-darker hover:border-clay/30 rounded-2xl p-4 text-left transition-all relative"
              >
                {hasUnreadAsync && (
                  <span className="absolute top-4 right-4 w-2 h-2 bg-rose rounded-full" />
                )}
                <div className="w-10 h-10 rounded-xl bg-clay-dim flex items-center justify-center flex-shrink-0">
                  <span className="font-display text-clay text-sm italic">M</span>
                </div>
                <div className="flex-1">
                  <p className="text-ink text-sm font-medium">Leave a message</p>
                  <p className="text-ink-ghost text-xs mt-0.5">Your partner will see it later</p>
                </div>
                <span className="text-ink-ghost group-hover:text-ink-dim transition"><ChevronRightIcon /></span>
              </button>
            </>
          ) : (
            /* No partner yet — show single pending card */
            <div className="w-full flex items-center gap-4 bg-parchment-deep/60 border border-dashed border-parchment-deeper rounded-2xl p-4">
              <div className="w-10 h-10 rounded-xl bg-parchment-deeper flex items-center justify-center flex-shrink-0">
                <span className="font-display text-ink-ghost text-sm italic">S</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-ink-muted text-sm font-medium">Waiting for your partner</p>
                <p className="text-ink-ghost text-xs mt-0.5">Share your invite code to get started</p>
                <button
                  onClick={e => { e.stopPropagation(); copyCode(activeCouple.invite_code, activeCouple.id) }}
                  className="mt-1.5 flex items-center gap-1.5 text-ink-ghost hover:text-ink-muted transition text-xs"
                >
                  {copied === activeCouple.id ? <CheckIcon /> : <CopyIcon />}
                  <span>{copied === activeCouple.id ? 'Copied!' : `Code: ${activeCouple.invite_code}`}</span>
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )

  // ── Sessions history ───────────────────────────────────────────────────────
  const SessionsHistory = () => (
    <div>
      {activeSessions.length > 0 && (
        <div className="mb-6">
          <p className="text-xs font-medium text-ink-ghost tracking-widest uppercase mb-3">In progress</p>
          <div className="space-y-2">
            {activeSessions.map(s => {
              const phase = PHASE_META[s.mediation_phase]
              return (
                <button key={s.id}
                  onClick={() => navigate(`/chat/${s.session_type}?couple_id=${activeCouple?.id}&session_id=${s.id}`)}
                  className="w-full flex items-center gap-3 bg-white/70 hover:bg-white border border-parchment-darker hover:border-terra/20 rounded-2xl px-4 py-3.5 text-left transition-all group"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <p className="text-ink text-sm font-medium">
                        {s.session_type === 'shared' ? `Shared · ${activeCouple?.partner || 'partner'}` :
                         s.session_type === 'individual' ? 'Individual' : 'Message'}
                      </p>
                      {phase && s.session_type === 'shared' && (
                        <span className="flex items-center gap-1">
                          <span className={`w-1.5 h-1.5 rounded-full ${phase.dot}`} />
                          <span className="text-ink-ghost text-2xs">{phase.label}</span>
                        </span>
                      )}
                    </div>
                    <p className="text-ink-ghost text-xs truncate">{s.last_message || 'Continue where you left off'}</p>
                  </div>
                  <span className="w-2 h-2 rounded-full bg-sage flex-shrink-0 animate-breathe" />
                </button>
              )
            })}
          </div>
        </div>
      )}

      {sessionsLoading && pastSessions.length === 0 && (
        <div className="space-y-2">
          {[1, 2, 3].map(i => (
            <div key={i} className="h-16 rounded-2xl skeleton" />
          ))}
        </div>
      )}

      {pastSessions.length > 0 && (
        <div>
          <p className="text-xs font-medium text-ink-ghost tracking-widest uppercase mb-3">Recent sessions</p>
          <div className="space-y-2">
            {pastSessions.map(s => {
              const phase = PHASE_META[s.mediation_phase]
              return (
                <button key={s.id}
                  onClick={() => navigate(`/chat/${s.session_type}?couple_id=${activeCouple?.id}&session_id=${s.id}`)}
                  className="w-full flex items-start gap-3 bg-white/40 hover:bg-white/70 border border-parchment-deeper hover:border-parchment-darker rounded-2xl px-4 py-3.5 text-left transition-all group"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-0.5">
                      <p className="text-ink-muted text-sm font-medium">
                        {s.session_type === 'shared' ? `Shared · ${activeCouple?.partner || 'partner'}` :
                         s.session_type === 'individual' ? 'Individual' : 'Message'}
                      </p>
                      {phase && s.session_type === 'shared' && (
                        <span className="flex items-center gap-1">
                          <span className={`w-1.5 h-1.5 rounded-full ${phase.dot} opacity-60`} />
                          <span className="text-ink-ghost text-2xs">{phase.label}</span>
                        </span>
                      )}
                    </div>
                    <p className="text-ink-ghost text-xs truncate">{s.last_message || 'Session ended'}</p>
                  </div>
                  <p className="text-ink-ghost text-2xs flex-shrink-0 mt-0.5">{relativeTime(s.created_at)}</p>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {!sessionsLoading && pastSessions.length === 0 && activeSessions.length === 0 && (
        <div className="text-center py-10">
          <p className="font-display text-ink-ghost text-base italic">No sessions yet</p>
          <p className="text-ink-ghost text-xs mt-1">Start a session to see your history here</p>
        </div>
      )}
    </div>
  )

  return (
    <div className="min-h-screen bg-parchment">

      {/* Desktop layout */}
      <div className="hidden lg:flex h-screen">

        {/* Left sidebar */}
        <div className="w-72 xl:w-80 flex flex-col bg-parchment-warm border-r border-parchment-deeper overflow-y-auto">
          {/* Logo + user */}
          <div className="px-6 pt-8 pb-6 border-b border-parchment-deeper">
            <div className="flex items-center justify-between mb-6">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-xl bg-terra-dim border border-terra/20 flex items-center justify-center">
                  <span className="font-display text-terra text-sm italic">B</span>
                </div>
                <span className="font-display text-ink text-lg font-medium">BOND</span>
              </div>
              <div className="flex items-center gap-1">
                <button onClick={() => navigate('/profile')} className="w-8 h-8 rounded-xl flex items-center justify-center text-ink-ghost hover:text-ink-muted hover:bg-parchment-deep transition">
                  <SettingsIcon />
                </button>
                <button onClick={() => setShowSignOutConfirm(true)} className="w-8 h-8 rounded-xl flex items-center justify-center text-ink-ghost hover:text-ink-muted hover:bg-parchment-deep transition">
                  <LogOutIcon />
                </button>
              </div>
            </div>
            <div>
              <p className="text-ink-ghost text-xs">{timeOfDay()}</p>
              <h1 className="font-display text-ink text-2xl font-medium">{name}</h1>
            </div>
          </div>

          <div className="flex-1 px-6 py-6 overflow-y-auto">
            <SidebarContent />
          </div>
        </div>

        {/* Main area */}
        <div className="flex-1 flex flex-col overflow-hidden">

          {/* Top bar */}
          <div className="px-8 py-6 border-b border-parchment-deeper flex items-center justify-between bg-parchment/80 backdrop-blur-sm">
            <div>
              <h2 className="font-display text-ink text-xl font-medium">
                {activeCouple ? activeCouple.label : 'Your space'}
              </h2>
              {activeCouple?.partner && (
                <p className="text-ink-ghost text-sm mt-0.5">with {activeCouple.partner}</p>
              )}
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto">
            <div className="max-w-2xl mx-auto px-8 py-8 space-y-10">

              {/* Actions */}
              <div>
                <p className="text-xs font-medium text-ink-ghost tracking-widest uppercase mb-4">Start a session</p>
                <SessionActions />
              </div>

              {/* Divider */}
              <div className="divider" />

              {/* History */}
              <SessionsHistory />
            </div>
          </div>
        </div>
      </div>

      {/* Mobile layout */}
      <div className="lg:hidden flex flex-col min-h-screen">

        {/* Mobile header */}
        <div className="bg-parchment-warm border-b border-parchment-deeper px-5 pt-10 pb-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-xl bg-terra-dim border border-terra/20 flex items-center justify-center">
                <span className="font-display text-terra text-sm italic">B</span>
              </div>
              <span className="font-display text-ink text-lg font-medium">BOND</span>
            </div>
            <div className="flex items-center gap-1">
              <button onClick={() => navigate('/profile')} className="w-9 h-9 rounded-xl flex items-center justify-center text-ink-ghost hover:text-ink-muted transition">
                <SettingsIcon />
              </button>
              <button onClick={() => setShowSignOutConfirm(true)} className="w-9 h-9 rounded-xl flex items-center justify-center text-ink-ghost hover:text-ink-muted transition">
                <LogOutIcon />
              </button>
            </div>
          </div>
          <div>
            <p className="text-ink-ghost text-xs">{timeOfDay()}</p>
            <h1 className="font-display text-ink text-2xl font-medium">{name}</h1>
          </div>

          {/* Mobile tabs */}
          <div className="flex gap-1 mt-4 bg-parchment-deep rounded-xl p-1">
            {[
              { id: 'home', label: 'Home' },
              { id: 'sessions', label: 'Sessions' },
              { id: 'connections', label: 'Connections' },
            ].map(tab => (
              <button key={tab.id} onClick={() => setMobilePanel(tab.id)}
                className={`flex-1 py-2 rounded-lg text-xs font-medium transition ${
                  mobilePanel === tab.id ? 'bg-white text-ink shadow-soft' : 'text-ink-ghost'
                }`}>
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Mobile panels */}
        <div className="flex-1 px-5 py-6 overflow-y-auto">
          {mobilePanel === 'home' && (
            <div className="space-y-6 animate-fade-up">
              <SessionActions />
            </div>
          )}
          {mobilePanel === 'sessions' && (
            <div className="animate-fade-up">
              <SessionsHistory />
            </div>
          )}
          {mobilePanel === 'connections' && (
            <div className="animate-fade-up">
              <SidebarContent />
            </div>
          )}
        </div>
      </div>

      {showSignOutConfirm && (
        <div className="fixed inset-0 bg-ink/20 backdrop-blur-sm z-50 flex items-center justify-center px-6">
          <div className="bg-parchment rounded-3xl p-6 w-full max-w-sm shadow-soft animate-fade-up">
            <p className="font-display text-ink text-lg font-medium mb-1">Sign out?</p>
            <p className="text-ink-dim text-sm mb-6 leading-relaxed">Your sessions and conversations are saved and will be here when you return.</p>
            <div className="flex gap-3">
              <button onClick={() => setShowSignOutConfirm(false)}
                className="flex-1 py-3 rounded-2xl border border-parchment-deeper text-ink-muted text-sm font-medium hover:bg-parchment-deep transition">
                Cancel
              </button>
              <button onClick={() => { localStorage.clear(); navigate('/login') }}
                className="flex-1 py-3 rounded-2xl bg-ink text-parchment text-sm font-medium hover:bg-ink-soft transition">
                Sign out
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
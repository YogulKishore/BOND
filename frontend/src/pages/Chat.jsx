import React, { useState, useEffect, useRef } from 'react'
import { useParams, useSearchParams } from 'react-router-dom'
import axios from 'axios'
import BASE from '../lib/api'
import useTransitionNavigate from '../hooks/useTransitionNavigate'

const WS_BASE = BASE.replace(/^http/, 'ws')

const PHASE_CONFIG = {
  listening:    { label: 'Listening',    color: 'text-sage',  bar: 'bg-sage' },
  understanding:{ label: 'Understanding',color: 'text-clay',  bar: 'bg-clay' },
  bridging:     { label: 'Bridging',     color: 'text-terra', bar: 'bg-terra' },
  resolution:   { label: 'Resolution',   color: 'text-terra', bar: 'bg-terra' },
  integration:  { label: 'Integration',  color: 'text-terra', bar: 'bg-terra' },
}

const MOODS = [
  { score: 1, emoji: '😔', label: 'distressed' },
  { score: 2, emoji: '😟', label: 'low' },
  { score: 3, emoji: '😐', label: 'neutral' },
  { score: 4, emoji: '🙂', label: 'okay' },
  { score: 5, emoji: '😊', label: 'good' },
]

function ArrowLeftIcon() {
  return (
    <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <polyline points="15 18 9 12 15 6"/>
    </svg>
  )
}
function SendIcon() {
  return (
    <svg width="15" height="15" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  )
}
function SparkleIcon() {
  return (
    <svg width="12" height="12" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
    </svg>
  )
}
function UsersIcon() {
  return (
    <svg width="13" height="13" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/>
      <path d="M23 21v-2a4 4 0 00-3-3.87"/><path d="M16 3.13a4 4 0 010 7.75"/>
    </svg>
  )
}
function XIcon() {
  return (
    <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
    </svg>
  )
}

export default function Chat() {
  const { sessionType } = useParams()
  const [searchParams] = useSearchParams()
  const navigate = useTransitionNavigate()
  const coupleId = searchParams.get('couple_id') || localStorage.getItem('couple_id') || 'demo'
  const token = localStorage.getItem('token')
  const name = localStorage.getItem('name')

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [partnerOnline, setPartnerOnline] = useState(false)
  const sessionStorageKey = `session_${sessionType}_${coupleId}`
  const [realSessionId, setRealSessionId] = useState(
    searchParams.get('session_id') || localStorage.getItem(sessionStorageKey) || null
  )
  const realSessionIdRef = useRef(
    searchParams.get('session_id') || localStorage.getItem(sessionStorageKey) || null
  )
  const [showEndConfirm, setShowEndConfirm] = useState(false)
  const [mediationPhase, setMediationPhase] = useState('listening')
  const mediationPhaseRef = useRef('listening')
  const [showConsentPrompt, setShowConsentPrompt] = useState(false)
  const [consentPending, setConsentPending] = useState(false)
  const [sendError, setSendError] = useState(false)

  const sessionStorageKeyEarly = `session_${sessionType}_${searchParams.get('couple_id') || localStorage.getItem('couple_id') || 'demo'}`
  const isResumingSession = !!searchParams.get('session_id') || !!localStorage.getItem(sessionStorageKeyEarly)
  const [showMoodCheck, setShowMoodCheck] = useState(sessionType === 'individual' && !isResumingSession)
  const [moodSelected, setMoodSelected] = useState(null)
  const [moodIntention, setMoodIntention] = useState('')
  const [savingMood, setSavingMood] = useState(false)

  const [sessionId] = useState(() => {
    const url = searchParams.get('session_id')
    if (url) return url
    if (sessionType === 'shared') return `shared_${coupleId}`
    return `${sessionType}_${localStorage.getItem('user_id')}_${Date.now()}`
  })

  const wsRef = useRef(null)
  const bottomRef = useRef(null)
  const sessionEndedRef = useRef(false)
  const inputRef = useRef(null)

  const triggerSessionEnd = (sid) => {
    if (!sid || sessionEndedRef.current) return
    sessionEndedRef.current = true
    fetch(`${BASE}/sessions/${sid}/end?token=${token}`, { method: 'GET', keepalive: true }).catch(() => {})
  }

  // Tracks whether the tab is being closed/navigated away vs just reloaded.
  // On reload, the page briefly becomes hidden then immediately visible again.
  // We use a short debounce so a real close (stays hidden) ends the session,
  // but a reload (visible again within ~500ms) does not.
  const _endTimerRef = React.useRef(null)

  const submitMoodCheck = async () => {
    if (!moodSelected) return
    setSavingMood(true)
    try {
      await axios.post(`${BASE}/profile/checkin`, {
        token, couple_id: coupleId,
        mood_score: moodSelected.score, mood_label: moodSelected.label,
        intention: moodIntention.trim(), session_type: 'individual',
      })
    } catch (e) {}
    setSavingMood(false)
    setShowMoodCheck(false)
  }

  const sendConsent = (agreed) => {
    setShowConsentPrompt(false)
    setConsentPending(true)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'bridge_consent', consent: agreed }))
    }
    setMessages(prev => [...prev, { role: 'user', sender: name, content: agreed ? "Yes, I'm open to hearing it." : "Not right now." }])
  }

  // Lock body scroll while in chat — restored on unmount
  useEffect(() => {
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => { document.body.style.overflow = prev }
  }, [])

  useEffect(() => {
    const initSession = async () => {
      if (sessionType === 'shared') {
        let wsSessionId = searchParams.get('session_id')
        if (!wsSessionId) {
          try {
            const res = await axios.post(`${BASE}/sessions/create?couple_id=${coupleId}&session_type=shared&token=${token}`)
            wsSessionId = res.data.session_id
            if (res.data.mediation_phase) {
              mediationPhaseRef.current = res.data.mediation_phase
              setMediationPhase(res.data.mediation_phase)
            }
            // Only wipe the stored key if the backend gave us a brand-new session
            const storedId = localStorage.getItem(sessionStorageKey)
            if (storedId && storedId !== wsSessionId) {
              localStorage.removeItem(sessionStorageKey)
            }
          } catch { wsSessionId = localStorage.getItem(sessionStorageKey) || `shared_${coupleId}` }
        }

        // Pre-load history via REST so messages show immediately on reload
        // (WS history event arrives async and can race with the blank state)
        try {
          const histRes = await axios.get(`${BASE}/sessions/history/${wsSessionId}?token=${token}`)
          const histMsgs = histRes.data.messages || []
          if (histMsgs.length > 0) {
            setMessages(histMsgs.map(m => ({
              role: m.sender_id === 'ai' ? 'ai' : 'user',
              sender: m.sender_id === 'ai' ? 'BOND' : name,
              content: m.content,
            })))
          }
        } catch {}
        setRealSessionId(wsSessionId)
        realSessionIdRef.current = wsSessionId
        localStorage.setItem(sessionStorageKey, wsSessionId)

        const ws = new WebSocket(`${WS_BASE}/ws/shared/${wsSessionId}?token=${token}`)
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data)
          if (data.type === 'history') {
            setMessages(data.messages.length === 0 ? [] : data.messages)
            if (data.mediation_phase) { mediationPhaseRef.current = data.mediation_phase; setMediationPhase(data.mediation_phase) }
          } else if (data.type === 'partner_status') {
            setPartnerOnline(data.online)
          } else if (data.type === 'typing') {
            setLoading(true)
          } else if (data.type === 'message') {
            setLoading(false); setConsentPending(false)
            setMessages(prev => [...prev, { ...data, role: 'ai', sender: 'BOND' }])
            if (mediationPhaseRef.current === 'bridging') {
              const text = data.content?.toLowerCase() || ''
              if (text.includes('would you be open') || text.includes('open to hearing')) setShowConsentPrompt(true)
            }
          } else if (data.type === 'bridge') {
            setLoading(false)
            setMessages(prev => [...prev, { ...data, role: 'bridge' }])
          } else if (data.type === 'resolution') {
            setLoading(false); setConsentPending(false)
            setMessages(prev => [...prev, { ...data, role: 'resolution' }])
          } else if (data.type === 'closing') {
            setLoading(false)
            setMessages(prev => [...prev, { ...data, role: 'closing' }])
          } else if (data.type === 'phase_change') {
            mediationPhaseRef.current = data.phase; setMediationPhase(data.phase)
            if (data.phase === 'resolution') setMessages(prev => [...prev, { role: 'phase', content: 'BOND is ready to share what it sees' }])
          }
        }
        ws.onerror = () => {}
        wsRef.current = ws
      } else {
        const urlSessionId = searchParams.get('session_id')
        const storedSessionId = localStorage.getItem(sessionStorageKey)
        const resumeSessionId = urlSessionId || storedSessionId

        if (resumeSessionId) {
          setRealSessionId(resumeSessionId); realSessionIdRef.current = resumeSessionId
          localStorage.setItem(sessionStorageKey, resumeSessionId)
          try {
            const res = await axios.get(`${BASE}/sessions/history/${resumeSessionId}?token=${token}`)
            setMessages(res.data.messages?.length > 0
              ? res.data.messages.map(m => ({ role: m.sender_id === 'ai' ? 'ai' : 'user', sender: m.sender_id === 'ai' ? 'BOND' : name, content: m.content }))
              : [])
          } catch {
            setMessages([])
          }
        } else {
          setMessages([])
        }
      }
    }
    initSession()
    const handleVisibility = () => {
      if (document.visibilityState === 'hidden') {
        // Start a short timer — if still hidden after 600ms, treat as real close
        _endTimerRef.current = setTimeout(() => {
          triggerSessionEnd(realSessionIdRef.current || sessionId)
        }, 600)
      } else {
        // Became visible again (reload) — cancel the end
        if (_endTimerRef.current) clearTimeout(_endTimerRef.current)
      }
    }
    document.addEventListener('visibilitychange', handleVisibility)
    return () => {
      wsRef.current?.close()
      document.removeEventListener('visibilitychange', handleVisibility)
      if (_endTimerRef.current) clearTimeout(_endTimerRef.current)
    }
  }, [])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, showConsentPrompt])

  const sendMessage = async () => {
    if (!input.trim() || loading) return
    const userMsg = input.trim()
    setInput(''); setSendError(false)
    setMessages(prev => [...prev, { role: 'user', sender: name, content: userMsg }])
    inputRef.current?.focus()

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ message: userMsg }))
      setLoading(true)
    } else {
      setLoading(true)
      try {
        const res = await axios.post(`${BASE}/chat/message`, {
          message: userMsg, session_type: sessionType, speaker_name: name,
          couple_id: coupleId, session_id: realSessionId || '', token, history: messages.slice(-10)
        })
        setMessages(prev => [...prev, { role: 'ai', sender: 'BOND', content: res.data.response }])
        if (res.data.session_id) { setRealSessionId(res.data.session_id); realSessionIdRef.current = res.data.session_id; localStorage.setItem(sessionStorageKey, res.data.session_id) }
      } catch {
        setSendError(true)
        setMessages(prev => [...prev, { role: 'ai', sender: 'BOND', content: "I'm having a little trouble right now — give it a moment and try again." }])
      }
      setLoading(false)
    }
  }

  const handleEndSession = () => {
    const sid = realSessionId || sessionId
    if (sid && !sessionEndedRef.current) {
      sessionEndedRef.current = true
      // Fire and forget — don't await, navigate immediately
      axios.get(`${BASE}/sessions/${sid}/end?token=${token}`).catch(() => {})
    }
    wsRef.current?.close()
    localStorage.removeItem(sessionStorageKey)
    navigate('/dashboard')
  }

  const phaseConfig = PHASE_CONFIG[mediationPhase]

  // ── Mood check overlay ─────────────────────────────────────────────────────
  if (showMoodCheck) {
    return (
      <div className="min-h-screen bg-parchment flex flex-col items-center justify-center px-6 py-12">
        <div className="w-full max-w-sm animate-fade-up">
          <div className="text-center mb-8">
            <div className="w-8 h-0.5 bg-terra/40 rounded-full mx-auto mb-5" />
            <p className="font-display text-ink-dim text-lg italic">Before we start</p>
          </div>
          <div className="bg-white/70 rounded-3xl p-7 border border-parchment-darker shadow-soft">
            <h2 className="font-display text-ink text-xl font-medium mb-1">How are you feeling?</h2>
            <p className="text-ink-ghost text-xs mb-6">Helps BOND meet you where you are</p>
            <div className="flex justify-between gap-2 mb-6">
              {MOODS.map(m => (
                <button key={m.score} onClick={() => setMoodSelected(m)}
                  className={`flex-1 flex flex-col items-center gap-1.5 py-3 rounded-2xl border transition-all ${
                    moodSelected?.score === m.score ? 'bg-terra-dim border-terra/30 scale-105' : 'bg-parchment-warm/60 border-parchment-darker hover:border-parchment-deeper'
                  }`}>
                  <span className="text-xl">{m.emoji}</span>
                  <span className={`text-2xs ${moodSelected?.score === m.score ? 'text-terra' : 'text-ink-ghost'}`}>{m.score}</span>
                </button>
              ))}
            </div>
            {moodSelected && (
              <div className="animate-slide-up space-y-3">
                <textarea value={moodIntention} onChange={e => setMoodIntention(e.target.value)}
                  placeholder="Anything specific on your mind? (optional)" rows={2} autoFocus
                  className="w-full bg-parchment/60 border border-parchment-darker rounded-2xl px-4 py-3 text-ink placeholder-ink-ghost focus:outline-none focus:border-terra/30 resize-none text-sm transition leading-relaxed" />
                <button onClick={submitMoodCheck} disabled={savingMood}
                  className="w-full bg-ink hover:bg-ink-soft disabled:opacity-40 text-parchment font-medium py-3.5 rounded-2xl transition text-sm">
                  {savingMood ? <span className="flex items-center justify-center gap-2"><span className="w-4 h-4 border-2 border-parchment/20 border-t-parchment rounded-full animate-spin" /> Starting...</span> : 'Start session'}
                </button>
              </div>
            )}
          </div>
          <button onClick={() => setShowMoodCheck(false)} className="w-full text-center text-ink-ghost text-xs mt-5 hover:text-ink-faint transition py-2">
            Skip for now
          </button>
        </div>
      </div>
    )
  }

  // ── Main chat ──────────────────────────────────────────────────────────────
  return (
    <div className="h-screen bg-parchment flex flex-col max-w-2xl mx-auto lg:border-x lg:border-parchment-deeper overflow-hidden">

      {/* Phase progress bar */}
      {sessionType === 'shared' && phaseConfig && (
        <div className="h-0.5 w-full bg-parchment-deep">
          <div className={`h-full ${phaseConfig.bar} transition-all duration-1000`}
            style={{ width: { listening: '20%', understanding: '40%', bridging: '60%', resolution: '80%', integration: '95%' }[mediationPhase] || '20%' }} />
        </div>
      )}

      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-4 border-b border-parchment-deeper bg-parchment/95 backdrop-blur-sm sticky top-0 z-10">
        <button onClick={handleEndSession}
          className="w-9 h-9 rounded-xl bg-parchment-warm border border-parchment-deeper flex items-center justify-center text-ink-muted hover:text-ink transition">
          <ArrowLeftIcon />
        </button>

        <div className="flex-1 min-w-0">
          <h2 className="font-display text-ink text-base font-medium">
            {sessionType === 'shared' ? 'Shared session' : sessionType === 'individual' ? 'Your space' : 'Leave a message'}
          </h2>
          {sessionType === 'shared' && phaseConfig && (
            <p className={`text-xs ${phaseConfig.color} mt-0.5`}>{phaseConfig.label} phase</p>
          )}
        </div>

        <div className="flex items-center gap-2">
          {sessionType === 'shared' && (
            <div className={`flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded-xl border transition ${
              partnerOnline ? 'text-sage border-sage/20 bg-sage-dim' : 'text-ink-ghost border-parchment-darker bg-parchment-warm'
            }`}>
              <UsersIcon />
              <span>{partnerOnline ? 'Partner here' : 'Waiting'}</span>
              {partnerOnline && <span className="w-1.5 h-1.5 rounded-full bg-sage animate-breathe" />}
            </div>
          )}
          <button onClick={() => setShowEndConfirm(true)}
            className="w-9 h-9 rounded-xl bg-parchment-warm border border-parchment-deeper flex items-center justify-center text-ink-ghost hover:text-rose transition">
            <XIcon />
          </button>
        </div>
      </div>

      {/* End session confirm */}
      {showEndConfirm && (
        <div className="mx-5 mt-3 bg-white/80 border border-parchment-darker rounded-2xl p-4 shadow-soft animate-slide-up">
          <p className="font-display text-ink text-base font-medium mb-1">End this session?</p>
          <p className="text-ink-dim text-sm mb-4 leading-relaxed">The conversation will be saved and you can come back to it.</p>
          <div className="flex gap-2.5">
            <button onClick={handleEndSession} className="flex-1 bg-rose/15 hover:bg-rose/25 text-rose py-2.5 rounded-xl text-sm transition font-medium">
              End session
            </button>
            <button onClick={() => setShowEndConfirm(false)} className="flex-1 bg-parchment-warm hover:bg-parchment-deep text-ink-muted py-2.5 rounded-xl text-sm transition">
              Keep going
            </button>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-6">
        <div className="space-y-5 max-w-xl mx-auto">

          {messages.map((msg, i) => (
            <div key={i} className="animate-fade-up">

              {/* System / phase divider */}
              {(msg.role === 'system' || msg.role === 'phase') && (
                <div className="flex items-center gap-4 py-2 my-2">
                  <div className="flex-1 divider" />
                  <div className="flex items-center gap-1.5 text-ink-ghost text-xs">
                    {msg.role === 'phase' && <SparkleIcon />}
                    <span className="font-display italic">{msg.content}</span>
                    {msg.role === 'phase' && <SparkleIcon />}
                  </div>
                  <div className="flex-1 divider" />
                </div>
              )}

              {/* Bridge insight */}
              {msg.role === 'bridge' && (
                <div className="flex justify-center my-3">
                  <div className="max-w-xs bg-terra-dim border border-terra/20 rounded-2xl px-5 py-4 text-center">
                    <p className="text-terra text-2xs font-medium uppercase tracking-widest mb-2">BOND · A thought</p>
                    <p className="text-ink text-sm leading-relaxed font-display italic">{msg.content}</p>
                  </div>
                </div>
              )}

              {/* Resolution */}
              {msg.role === 'resolution' && (
                <div className="my-4">
                  <div className="bg-white/80 border border-parchment-darker rounded-2xl px-5 py-5 shadow-soft">
                    <div className="flex items-center gap-2 mb-3">
                      <SparkleIcon />
                      <p className="text-ink-dim text-2xs font-medium uppercase tracking-widest">BOND · What I'm seeing</p>
                    </div>
                    <p className="text-ink text-sm leading-relaxed font-display">{msg.content}</p>
                  </div>
                </div>
              )}

              {/* Closing */}
              {msg.role === 'closing' && (
                <div className="flex justify-center my-4">
                  <div className="max-w-xs bg-parchment-warm border border-parchment-deeper rounded-2xl px-5 py-4 text-center">
                    <p className="text-ink-ghost text-2xs font-medium uppercase tracking-widest mb-2">BOND · Before you go</p>
                    <p className="text-ink-muted text-sm leading-relaxed font-display italic">{msg.content}</p>
                  </div>
                </div>
              )}

              {/* Regular messages */}
              {!['system', 'phase', 'bridge', 'resolution', 'closing'].includes(msg.role) && (
                <div className={`flex flex-col ${msg.role === 'user' && msg.sender === name ? 'items-end' : 'items-start'}`}>
                  <span className="text-2xs text-ink-ghost mb-1.5 px-1">
                    {msg.role === 'ai' ? 'BOND' : msg.sender}
                  </span>
                  <div className={`max-w-sm lg:max-w-md text-sm leading-relaxed px-4 py-3 ${
                    msg.role === 'ai'
                      ? 'bg-white/80 border border-parchment-darker rounded-2xl rounded-tl-sm text-ink shadow-soft'
                      : msg.sender === name
                      ? 'bg-ink text-parchment rounded-2xl rounded-tr-sm shadow-soft'
                      : 'bg-parchment-warm border border-parchment-deeper rounded-2xl rounded-tl-sm text-ink'
                  }`}>
                    {msg.content}
                  </div>
                </div>
              )}

            </div>
          ))}

          {/* Typing indicator */}
          {loading && (
            <div className="flex flex-col items-start animate-fade-up">
              <span className="text-2xs text-ink-ghost mb-1.5 px-1">BOND</span>
              <div className="bg-white/80 border border-parchment-darker px-4 py-3 rounded-2xl rounded-tl-sm shadow-soft">
                <div className="flex gap-1.5 items-center">
                  <span className="w-1.5 h-1.5 bg-ink-ghost rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1.5 h-1.5 bg-ink-ghost rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1.5 h-1.5 bg-ink-ghost rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
              </div>
            </div>
          )}

          {/* Waiting for partner */}
          {consentPending && !loading && (
            <div className="flex items-center gap-4 py-1">
              <div className="flex-1 divider" />
              <p className="text-ink-ghost text-xs font-display italic">Waiting for your partner...</p>
              <div className="flex-1 divider" />
            </div>
          )}

          {/* Consent */}
          {showConsentPrompt && (
            <div className="flex flex-col items-center gap-3 my-3 animate-fade-up">
              <div className="flex gap-2.5 w-full max-w-xs">
                <button onClick={() => sendConsent(true)}
                  className="flex-1 bg-ink hover:bg-ink-soft text-parchment font-medium py-3 rounded-xl text-sm transition shadow-soft">
                  Yes, I'm open
                </button>
                <button onClick={() => sendConsent(false)}
                  className="flex-1 bg-parchment-warm hover:bg-parchment-deep border border-parchment-deeper text-ink-muted py-3 rounded-xl text-sm transition">
                  Not yet
                </button>
              </div>
              <p className="text-ink-ghost text-xs text-center max-w-xs leading-relaxed">
                BOND will share what it's observed — nothing your partner said directly
              </p>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input */}
      <div className="border-t border-parchment-deeper bg-parchment/95 backdrop-blur-sm px-5 py-4">
        {sendError && (
          <p className="text-rose text-xs text-center mb-2">Message failed — check your connection</p>
        )}
        <div className="flex gap-3 items-end max-w-xl mx-auto">
          <textarea
            ref={inputRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
            placeholder="Say what's on your mind..."
            rows={1}
            autoFocus
            className="flex-1 bg-white/70 border border-parchment-darker rounded-2xl px-4 py-3 text-ink placeholder-ink-ghost focus:outline-none focus:border-terra/30 resize-none text-sm transition leading-relaxed"
          />
          <button onClick={sendMessage} disabled={loading || !input.trim()}
            className="w-10 h-10 bg-ink hover:bg-ink-soft disabled:opacity-30 text-parchment rounded-xl flex items-center justify-center transition shadow-soft flex-shrink-0">
            <SendIcon />
          </button>
        </div>
        <p className="text-center text-ink-ghost text-2xs mt-2.5">
          Not a substitute for professional therapy
        </p>
      </div>
    </div>
  )
}
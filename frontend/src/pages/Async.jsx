import { useState, useEffect, useRef } from 'react'
import useTransitionNavigate from '../hooks/useTransitionNavigate'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios'
import BASE from '../lib/api'

function ArrowLeftIcon() {
  return <svg width="16" height="16" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><polyline points="15 18 9 12 15 6"/></svg>
}
function SendIcon() {
  return <svg width="14" height="14" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
}

export default function Async() {
  const [searchParams] = useSearchParams()
  const navigate = useTransitionNavigate()
  const coupleId = searchParams.get('couple_id')
  const token = localStorage.getItem('token')
  const userId = localStorage.getItem('user_id')

  // Get partner name from stored couples
  const partnerName = (() => {
    try {
      const couples = JSON.parse(localStorage.getItem('couples')) || []
      return couples.find(c => c.id === coupleId)?.partner || null
    } catch { return null }
  })()

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [summary, setSummary] = useState(null)
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [error, setError] = useState(null)
  const summaryFetchedRef = useRef(false)
  const markedReadRef = useRef(false)
  const bottomRef = useRef(null)

  useEffect(() => {
    fetchMessages()
    const interval = setInterval(fetchMessages, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const fetchMessages = async () => {
    try {
      const res = await axios.get(`${BASE}/async/messages/${coupleId}?token=${token}`)
      setMessages(res.data.messages || [])
      setError(null)
      if (!markedReadRef.current) {
        markedReadRef.current = true
        const readTs = JSON.parse(localStorage.getItem('async_read') || '{}')
        readTs[coupleId] = new Date().toISOString()
        localStorage.setItem('async_read', JSON.stringify(readTs))        
        if (res.data.messages?.length > 0 && res.data.has_unread && !summaryFetchedRef.current) {
          summaryFetchedRef.current = true
          fetchSummary()
        }
      }
    } catch { setError('Could not load messages') }
  }

  const fetchSummary = async () => {
    setLoadingSummary(true)
    try {
      const res = await axios.get(`${BASE}/async/summary/${coupleId}?token=${token}`)
      setSummary(res.data.summary)
    } catch {}
    setLoadingSummary(false)
  }

  const sendMessage = async () => {
    if (!input.trim() || sending) return
    const content = input.trim()
    setSending(true); setInput('')
    try {
      const res = await axios.post(`${BASE}/async/send`, { token, couple_id: coupleId, content })
      setMessages(prev => [...prev, res.data.message])
    } catch { setInput(content); setError('Failed to send — try again') }
    setSending(false)
  }

  return (
    <div className="min-h-screen bg-parchment flex flex-col max-w-2xl mx-auto lg:border-x lg:border-parchment-deeper">

      {/* Header */}
      <div className="flex items-center gap-3 px-5 py-4 border-b border-parchment-deeper bg-parchment/95 backdrop-blur-sm">
        <button onClick={() => navigate('/dashboard')} className="w-9 h-9 rounded-xl bg-parchment-warm border border-parchment-deeper flex items-center justify-center text-ink-muted hover:text-ink transition">
          <ArrowLeftIcon />
        </button>
        <div>
          <h2 className="font-display text-ink text-base font-medium">
            {partnerName || 'Messages'}
          </h2>
          <p className="text-ink-ghost text-xs">Your partner will see this when they're ready</p>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-5 py-6 space-y-4">

        {error && (
          <div className="bg-rose-dim border border-rose/20 rounded-2xl px-4 py-3 text-center animate-slide-up">
            <p className="text-rose text-sm">{error}</p>
          </div>
        )}

        {loadingSummary && (
          <div className="bg-terra-dim border border-terra/20 rounded-2xl p-4">
            <p className="text-terra text-2xs font-medium uppercase tracking-widest mb-2">BOND</p>
            <div className="flex gap-1.5 items-center">
              {[0, 150, 300].map(delay => (
                <span key={delay} className="w-1.5 h-1.5 bg-terra/60 rounded-full animate-bounce" style={{ animationDelay: `${delay}ms` }} />
              ))}
            </div>
          </div>
        )}

        {summary && (
          <div className="bg-terra-dim border border-terra/20 rounded-2xl p-5 animate-slide-up">
            <p className="text-terra text-2xs font-medium uppercase tracking-widest mb-2">BOND · Summary</p>
            <p className="text-ink text-sm leading-relaxed font-display italic">{summary}</p>
          </div>
        )}

        {messages.length === 0 && !loadingSummary && (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="w-12 h-12 rounded-2xl bg-parchment-deep border border-parchment-deeper flex items-center justify-center mb-4">
              <SendIcon />
            </div>
            <p className="font-display text-ink-dim text-base italic mb-1">No messages yet</p>
            <p className="text-ink-ghost text-sm max-w-xs leading-relaxed">
              Leave something thoughtful for your partner to read when they're ready
            </p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={msg.id || i} className={`flex flex-col ${msg.sender_id === userId ? 'items-end' : 'items-start'} animate-fade-up`}>
            <span className="text-2xs text-ink-ghost mb-1.5 px-1">{msg.sender_name}</span>
            <div className={`max-w-xs px-4 py-3 rounded-2xl text-sm leading-relaxed ${
              msg.sender_id === userId
                ? 'bg-ink text-parchment rounded-tr-sm shadow-soft'
                : 'bg-white/80 border border-parchment-darker text-ink rounded-tl-sm shadow-soft'
            }`}>
              {msg.content}
            </div>
            <span className="text-2xs text-ink-ghost mt-1 px-1">
              {new Date(msg.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </span>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-parchment-deeper bg-parchment/95 backdrop-blur-sm px-5 py-4">
        <div className="flex gap-3 items-end">
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() } }}
            placeholder="Leave a thoughtful message..."
            rows={2}
            className="flex-1 bg-white/70 border border-parchment-darker rounded-2xl px-4 py-3 text-ink placeholder-ink-ghost focus:outline-none focus:border-terra/30 resize-none text-sm transition leading-relaxed"
          />
          <button onClick={sendMessage} disabled={sending || !input.trim()}
            className="w-10 h-10 bg-ink hover:bg-ink-soft disabled:opacity-30 text-parchment rounded-xl flex items-center justify-center transition shadow-soft flex-shrink-0">
            {sending
              ? <span className="w-4 h-4 border-2 border-parchment/20 border-t-parchment rounded-full animate-spin" />
              : <SendIcon />}
          </button>
        </div>
        <p className="text-ink-ghost text-2xs mt-2.5 text-center">
          BOND will summarize your messages for your partner
        </p>
      </div>
    </div>
  )
}
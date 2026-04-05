import { useState } from 'react'
import useTransitionNavigate from '../hooks/useTransitionNavigate'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios'
import BASE from '../lib/api'

const MOODS = [
  { score: 1, emoji: '😔', label: 'distressed', color: 'border-rose/40 bg-rose-dim' },
  { score: 2, emoji: '😟', label: 'low', color: 'border-clay/40 bg-clay-dim' },
  { score: 3, emoji: '😐', label: 'neutral', color: 'border-parchment-darker bg-parchment-warm' },
  { score: 4, emoji: '🙂', label: 'okay', color: 'border-sage/40 bg-sage-dim' },
  { score: 5, emoji: '😊', label: 'good', color: 'border-terra/40 bg-terra-dim' },
]

export default function CheckIn() {
  const [mood, setMood] = useState(null)
  const [intention, setIntention] = useState('')
  const [saving, setSaving] = useState(false)
  const navigate = useTransitionNavigate()
  const [searchParams] = useSearchParams()
  const coupleId = searchParams.get('couple_id')
  const sessionType = searchParams.get('session_type') || 'shared'

  const handleSubmit = async () => {
    if (!mood) return
    setSaving(true)
    try {
      await axios.post(`${BASE}/profile/checkin`, {
        token: localStorage.getItem('token'),
        couple_id: coupleId,
        mood_score: mood.score,
        mood_label: mood.label,
        intention: intention.trim(),
        session_type: sessionType,
      })
    } catch (e) { console.error(e) }
    navigate(`/chat/${sessionType}?couple_id=${coupleId}`)
  }

  const handleSkip = () => {
    navigate(`/chat/${sessionType}?couple_id=${coupleId}`)
  }

  return (
    <div className="min-h-screen bg-parchment flex flex-col items-center justify-center px-6 py-12">
      <div className="w-full max-w-sm animate-fade-up">

        <div className="text-center mb-10">
          <div className="w-8 h-0.5 bg-terra/40 rounded-full mx-auto mb-6" />
          <p className="font-display text-ink-dim text-lg italic">Before you start</p>
        </div>

        <div className="bg-white/60 rounded-3xl p-7 border border-parchment-deeper shadow-soft">
          <h2 className="font-display text-ink text-xl font-medium mb-1">How are you feeling?</h2>
          <p className="text-ink-ghost text-xs mb-6">Tap one — it helps BOND show up for you</p>

          <div className="flex justify-between gap-2 mb-8">
            {MOODS.map(m => (
              <button
                key={m.score}
                onClick={() => setMood(m)}
                className={`flex-1 flex flex-col items-center gap-1.5 py-3 rounded-2xl border transition-all duration-150 ${
                  mood?.score === m.score
                    ? m.color + ' scale-105'
                    : 'bg-parchment-warm/50 border-parchment-darker hover:border-parchment-deeper'
                }`}
              >
                <span className="text-xl">{m.emoji}</span>
                <span className={`text-2xs font-medium ${mood?.score === m.score ? 'text-ink-muted' : 'text-ink-ghost'}`}>
                  {m.label}
                </span>
              </button>
            ))}
          </div>

          {mood && (
            <div className="animate-slide-up space-y-3">
              <div>
                <p className="text-ink text-sm font-medium mb-2">Anything on your mind going in?</p>
                <textarea
                  value={intention}
                  onChange={e => setIntention(e.target.value)}
                  placeholder="Optional — anything you want BOND to keep in mind"
                  rows={3}
                  autoFocus
                  className="w-full bg-parchment/60 border border-parchment-darker rounded-2xl px-4 py-3 text-ink placeholder-ink-ghost focus:outline-none focus:border-terra/30 resize-none text-sm transition leading-relaxed"
                />
              </div>
              <button
                onClick={handleSubmit}
                disabled={saving}
                className="w-full bg-ink hover:bg-ink-soft disabled:opacity-40 text-parchment font-medium py-3.5 rounded-2xl transition text-sm shadow-soft"
              >
                {saving ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-parchment/20 border-t-parchment rounded-full animate-spin" />
                    Starting...
                  </span>
                ) : 'Start session'}
              </button>
            </div>
          )}
        </div>

        <button
          onClick={handleSkip}
          className="w-full text-center text-ink-ghost text-xs mt-5 hover:text-ink-faint transition py-2"
        >
          Skip for now
        </button>
      </div>
    </div>
  )
}
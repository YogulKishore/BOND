import { useState, useEffect } from 'react'
import useTransitionNavigate from '../hooks/useTransitionNavigate'
import { useSearchParams } from 'react-router-dom'
import axios from 'axios'
import BASE from '../lib/api'

const steps = [
  {
    key: 'duration',
    question: "How long have you two been together?",
    options: [
      { value: 'new', label: 'Less than 6 months' },
      { value: 'growing', label: '6 months to 2 years' },
      { value: 'established', label: '2 to 5 years' },
      { value: 'long', label: 'More than 5 years' },
    ]
  },
  {
    key: 'status',
    question: "How would you describe things between you right now?",
    options: [
      { value: 'great', label: "Things are good — just want extra support" },
      { value: 'rough', label: "Going through a rough patch" },
      { value: 'stuck', label: "Stuck in the same patterns" },
      { value: 'healing', label: "Recovering from something hard" },
    ]
  },
  {
    key: 'goal',
    question: "What would you like to change or improve right now?",
    options: [
      { value: 'communication', label: 'Communicate more openly' },
      { value: 'conflict', label: 'Handle conflict without escalating' },
      { value: 'reconnect', label: 'Reconnect and feel closer' },
      { value: 'understand', label: 'Understand each other better' },
    ]
  },
  {
    key: 'biggest_challenge',
    question: "What's been coming up between you two lately?",
    type: 'text',
    placeholder: "It's okay if it's hard to put into words...",
    hint: "The thing you keep circling back to — even if you can't quite name it."
  }
]

export default function RelationshipProfile() {
  const [step, setStep] = useState(0)
  const [answers, setAnswers] = useState({})
  const [selected, setSelected] = useState(null)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useTransitionNavigate()
  const [searchParams] = useSearchParams()
  const coupleId = searchParams.get('couple_id')
  const sessionType = searchParams.get('session_type')
  const current = steps[step]
  const isText = current.type === 'text'

  // If the couple is already profiled (partner already filled this in),
  // skip straight to check-in — second person should not redo this.
  useEffect(() => {
    const couples = (() => { try { return JSON.parse(localStorage.getItem('couples')) || [] } catch { return [] } })()
    const couple = couples.find(c => c.id === coupleId)
    if (couple?.is_relationship_profiled) {
      navigate(`/checkin?couple_id=${coupleId}&session_type=${sessionType}`, { replace: true })
    }
  }, [])

  const handleChoice = (value) => {
    setSelected(value)
    const newAnswers = { ...answers, [current.key]: value }
    setAnswers(newAnswers)
    setTimeout(() => {
      setSelected(null)
      if (step < steps.length - 1) setStep(step + 1)
      else submit(newAnswers)
    }, 200)
  }

  const handleText = () => {
    if (!text.trim()) return
    submit({ ...answers, [current.key]: text.trim() })
  }

  const submit = async (finalAnswers) => {
    setLoading(true)
    try {
      await axios.post(`${BASE}/profile/relationship`, {
        token: localStorage.getItem('token'),
        couple_id: coupleId,
        answers: finalAnswers
      })
    } catch (e) { console.error(e) }
    navigate(`/checkin?couple_id=${coupleId}&session_type=${sessionType}`)
  }

  return (
    <div className="min-h-screen bg-parchment flex flex-col">
      <div className="w-full h-0.5 bg-parchment-deep">
        <div
          className="h-full bg-sage transition-all duration-500 ease-out"
          style={{ width: `${((step + 1) / steps.length) * 100}%` }}
        />
      </div>

      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-lg animate-fade-up">

          {step === 0 && (
            <div className="mb-10 text-center">
              <p className="font-display text-ink-dim text-lg italic mb-2">About your connection</p>
              <p className="text-ink-dim text-sm leading-relaxed max-w-sm mx-auto">
                A few things to help BOND understand your relationship before you start together.
              </p>
            </div>
          )}

          <div className="mb-3">
            <span className="text-xs text-ink-ghost tracking-wide uppercase font-medium">
              {step + 1} of {steps.length}
            </span>
          </div>

          <h2 className="font-display text-ink text-2xl lg:text-3xl font-medium leading-snug mb-2 text-balance">
            {current.question}
          </h2>
          {current.hint && (
            <p className="text-ink-dim text-sm mb-8 leading-relaxed italic">{current.hint}</p>
          )}
          {!current.hint && <div className="mb-8" />}

          {!isText && (
            <div className="space-y-3">
              {current.options.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => handleChoice(opt.value)}
                  className={`w-full text-left px-5 py-4 rounded-2xl border text-sm transition-all duration-150 ${
                    selected === opt.value
                      ? 'bg-sage text-white border-sage shadow-soft'
                      : 'bg-white/50 hover:bg-white/80 border-parchment-darker hover:border-sage/40 text-ink'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          )}

          {isText && (
            <div className="space-y-3">
              <textarea
                value={text}
                onChange={e => setText(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleText() } }}
                placeholder={current.placeholder}
                rows={4}
                autoFocus
                className="w-full bg-white/60 border border-parchment-darker rounded-2xl px-4 py-4 text-ink placeholder-ink-ghost focus:outline-none focus:border-sage/40 resize-none text-sm transition leading-relaxed"
              />
              <button
                onClick={handleText}
                disabled={!text.trim() || loading}
                className="w-full bg-ink hover:bg-ink-soft disabled:opacity-40 disabled:cursor-not-allowed text-parchment font-medium py-3.5 rounded-2xl transition text-sm shadow-soft"
              >
                {loading ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-parchment/20 border-t-parchment rounded-full animate-spin" />
                    Saving...
                  </span>
                ) : 'Start session'}
              </button>
            </div>
          )}

          <p className="text-center text-ink-ghost text-xs mt-8">
            Only shown once per connection
          </p>
        </div>
      </div>
    </div>
  )
}
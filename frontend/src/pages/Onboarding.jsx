import { useState } from 'react'
import useTransitionNavigate from '../hooks/useTransitionNavigate'
import axios from 'axios'
import BASE from '../lib/api'

const steps = [
  {
    key: 'communication_style',
    question: "When something bothers you, what do you usually do?",
    hint: "There's no right answer — just what's true for you.",
    options: [
      { value: 'talk', label: 'Talk about it right away' },
      { value: 'withdraw', label: 'Need space before I can discuss it' },
      { value: 'indirect', label: 'Drop hints instead of saying it directly' },
      { value: 'suppress', label: 'Keep it to myself' },
    ]
  },
  {
    key: 'love_language',
    question: "What makes you feel most loved?",
    hint: "Think about moments when you felt genuinely cared for.",
    options: [
      { value: 'words', label: 'Hearing it — words of affirmation' },
      { value: 'time', label: 'Time — their full presence' },
      { value: 'acts', label: 'Actions — them doing things for me' },
      { value: 'touch', label: 'Physical closeness and touch' },
    ]
  },
  {
    key: 'conflict_style',
    question: "During tense moments, you tend to...",
    hint: "Be honest — this helps BOND understand how to support you.",
    options: [
      { value: 'escalate', label: 'Get heated quickly' },
      { value: 'shutdown', label: 'Go quiet and withdraw' },
      { value: 'deflect', label: 'Use humor to deflect' },
      { value: 'resolve', label: 'Push to resolve it fast' },
    ]
  },
  {
    key: 'support_style',
    question: "When you're struggling, what helps most?",
    hint: "What actually helps you when you're struggling?",
    options: [
      { value: 'listen', label: 'Being heard without advice' },
      { value: 'advice', label: 'Practical solutions and ideas' },
      { value: 'reframe', label: 'A different way of seeing it' },
      { value: 'space', label: 'Quiet space to process alone' },
    ]
  },
  {
    key: 'hope',
    question: "What would you want to feel differently in your relationship?",
    hint: "Say it in your own words. It could be anything.",
    type: 'text',
    placeholder: "e.g. understand why we keep fighting about the same things..."
  }
]

export default function Onboarding() {
  const saved = JSON.parse(localStorage.getItem('onboarding_progress') || '{}')
  const [step, setStep] = useState(saved.step || 0)
  const [answers, setAnswers] = useState(saved.answers || {})
  const [selected, setSelected] = useState(null)
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const navigate = useTransitionNavigate()
  const name = localStorage.getItem('name')
  const current = steps[step]
  const isText = current.type === 'text'

  const handleChoice = (value) => {
    setSelected(value)
    const newAnswers = { ...answers, [current.key]: value }
    setAnswers(newAnswers)
    localStorage.setItem('onboarding_progress', JSON.stringify({ step: step + 1, answers: newAnswers }))
    setTimeout(() => {
      setSelected(null)
      if (step < steps.length - 1) setStep(step + 1)
      else submit(newAnswers)
    }, 200)
  }

  const handleText = () => {
    if (!text.trim()) return
    const newAnswers = { ...answers, [current.key]: text.trim() }
    localStorage.setItem('onboarding_progress', JSON.stringify({ step: step + 1, answers: newAnswers }))
    submit(newAnswers)
  }

  const submit = async (finalAnswers) => {
    setLoading(true)
    try {
      await axios.post(`${BASE}/profile/onboarding`, {
        token: localStorage.getItem('token'),
        answers: finalAnswers
      })
    } catch (e) { console.error(e) }
    localStorage.removeItem('onboarding_progress')
    localStorage.setItem('onboarded', 'true')
    navigate('/dashboard')
  }

  return (
    <div className="min-h-screen bg-parchment flex flex-col">

      {/* Progress bar */}
      <div className="w-full h-0.5 bg-parchment-deep">
        <div
          className="h-full bg-terra transition-all duration-500 ease-out"
          style={{ width: `${((step + 1) / steps.length) * 100}%` }}
        />
      </div>

      <div className="flex-1 flex items-center justify-center px-6 py-12">
        <div className="w-full max-w-lg animate-fade-up">

          {step === 0 && (
            <div className="mb-10 text-center">
              <p className="font-display text-ink-dim text-lg italic mb-2">Hi, {name}</p>
              <p className="text-ink-dim text-sm leading-relaxed max-w-sm mx-auto">
                A few questions so BOND can understand how you communicate and what you need. Completely private.
              </p>
            </div>
          )}

          <div className="mb-3 flex items-center justify-between">
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

          {!isText && (
            <div className="space-y-3">
              {current.options.map(opt => (
                <button
                  key={opt.value}
                  onClick={() => handleChoice(opt.value)}
                  className={`w-full text-left px-5 py-4 rounded-2xl border text-sm transition-all duration-150 ${
                    selected === opt.value
                      ? 'bg-terra text-parchment border-terra shadow-warm'
                      : 'bg-white/50 hover:bg-white/80 border-parchment-darker hover:border-clay/40 text-ink'
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
                className="w-full bg-white/60 border border-parchment-darker rounded-2xl px-4 py-4 text-ink placeholder-ink-ghost focus:outline-none focus:border-terra/40 resize-none text-sm transition leading-relaxed"
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
                ) : 'Finish setup'}
              </button>
            </div>
          )}

          <p className="text-center text-ink-ghost text-xs mt-8">
            Your answers are never shared with your partner
          </p>
        </div>
      </div>
    </div>
  )
}
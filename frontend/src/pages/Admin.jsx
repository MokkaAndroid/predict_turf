import { useState } from 'react'
import { postCollect, postCollectToday, postTrain, postPredict, postBacktest, getHealth } from '../api'
import { Settings, Download, Calendar, Brain, Target, BarChart3, Activity, Loader2, CheckCircle2, XCircle, BookOpen } from 'lucide-react'

function ActionButton({ label, description, onClick, loading, result, icon: Icon, accent = 'racing' }) {
  const accents = {
    racing: 'bg-racing-600 hover:bg-racing-500 shadow-md hover:shadow-lg',
    gold: 'bg-gradient-to-r from-gold-500 to-gold-400 text-racing-950 hover:from-gold-400 hover:to-gold-300 shadow-gold',
    blue: 'bg-blue-600 hover:bg-blue-500 shadow-md hover:shadow-lg',
    purple: 'bg-purple-600 hover:bg-purple-500 shadow-md hover:shadow-lg',
  }

  return (
    <div className="card-glass p-5 group hover:border-dark-600/50 transition-all duration-300">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-xl bg-dark-800 flex items-center justify-center flex-shrink-0">
          <Icon className="w-4.5 h-4.5 text-dark-300" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-white text-sm">{label}</h3>
          <p className="text-xs text-dark-400 mt-1 leading-relaxed">{description}</p>
        </div>
      </div>
      <button
        onClick={onClick}
        disabled={loading}
        className={`mt-4 w-full px-4 py-2.5 rounded-xl text-white text-sm font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${accents[accent]} disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0`}
      >
        {loading ? (
          <>
            <Loader2 className="w-4 h-4 animate-spin" />
            En cours...
          </>
        ) : (
          'Lancer'
        )}
      </button>
      {result && (
        <div className="mt-3 relative">
          <div className={`absolute top-2 right-2 ${result.error ? 'text-red-400' : 'text-racing-400'}`}>
            {result.error ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
          </div>
          <pre className="bg-dark-800/50 border border-dark-700/30 rounded-xl p-3 text-[11px] text-dark-300 overflow-auto max-h-40 font-mono">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  )
}

export default function Admin() {
  const [loading, setLoading] = useState({})
  const [results, setResults] = useState({})
  const [startDate, setStartDate] = useState('')
  const [endDate, setEndDate] = useState('')

  const run = async (key, fn) => {
    setLoading(prev => ({ ...prev, [key]: true }))
    try {
      const result = await fn()
      setResults(prev => ({ ...prev, [key]: result }))
    } catch (err) {
      setResults(prev => ({ ...prev, [key]: { error: err.message } }))
    }
    setLoading(prev => ({ ...prev, [key]: false }))
  }

  return (
    <div className="space-y-8 animate-slide-up">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-dark-800 flex items-center justify-center">
          <Settings className="w-5 h-5 text-dark-300" />
        </div>
        <div>
          <h1 className="text-3xl font-display font-bold text-white">Administration</h1>
          <p className="text-dark-400 text-sm mt-0.5">Gestion du pipeline de donnees et du modele</p>
        </div>
      </div>

      {/* Actions Grid */}
      <div className="grid md:grid-cols-2 gap-4">
        <ActionButton
          label="Collecter aujourd'hui"
          description="Recupere les courses de plat du jour depuis l'API PMU."
          onClick={() => run('today', postCollectToday)}
          loading={loading.today}
          result={results.today}
          icon={Download}
          accent="racing"
        />

        {/* Custom period collector */}
        <div className="card-glass p-5 group hover:border-dark-600/50 transition-all duration-300">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-xl bg-dark-800 flex items-center justify-center flex-shrink-0">
              <Calendar className="w-4.5 h-4.5 text-dark-300" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-white text-sm">Collecter une periode</h3>
              <p className="text-xs text-dark-400 mt-1 leading-relaxed">Recupere l'historique des courses sur une plage de dates.</p>
            </div>
          </div>
          <div className="flex gap-2 mt-4">
            <input
              type="date"
              value={startDate}
              onChange={e => setStartDate(e.target.value)}
              className="flex-1 bg-dark-800 border border-dark-700 rounded-xl px-3 py-2.5 text-sm text-dark-200 focus:outline-none focus:border-gold-400/50 focus:ring-1 focus:ring-gold-400/20 transition-colors"
            />
            <input
              type="date"
              value={endDate}
              onChange={e => setEndDate(e.target.value)}
              className="flex-1 bg-dark-800 border border-dark-700 rounded-xl px-3 py-2.5 text-sm text-dark-200 focus:outline-none focus:border-gold-400/50 focus:ring-1 focus:ring-gold-400/20 transition-colors"
            />
          </div>
          <button
            onClick={() => run('range', () => postCollect(startDate, endDate))}
            disabled={loading.range || !startDate || !endDate}
            className="mt-3 w-full px-4 py-2.5 rounded-xl text-white text-sm font-semibold bg-racing-600 hover:bg-racing-500 shadow-md hover:shadow-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 flex items-center justify-center gap-2"
          >
            {loading.range ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                En cours...
              </>
            ) : (
              'Collecter'
            )}
          </button>
          {results.range && (
            <div className="mt-3 relative">
              <div className={`absolute top-2 right-2 ${results.range.error ? 'text-red-400' : 'text-racing-400'}`}>
                {results.range.error ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
              </div>
              <pre className="bg-dark-800/50 border border-dark-700/30 rounded-xl p-3 text-[11px] text-dark-300 overflow-auto max-h-40 font-mono">
                {JSON.stringify(results.range, null, 2)}
              </pre>
            </div>
          )}
        </div>

        <ActionButton
          label="Entrainer le modele"
          description="Entraine le modele ML (LightGBM) sur toutes les courses terminees. Walk-forward validation."
          onClick={() => run('train', postTrain)}
          loading={loading.train}
          result={results.train}
          icon={Brain}
          accent="blue"
        />

        <ActionButton
          label="Generer les predictions"
          description="Genere les predictions pour toutes les courses a venir non encore predites."
          onClick={() => run('predict', postPredict)}
          loading={loading.predict}
          result={results.predict}
          icon={Target}
          accent="gold"
        />

        <ActionButton
          label="Backtester"
          description="Compare les predictions aux resultats reels et calcule les gains simules."
          onClick={() => run('backtest', postBacktest)}
          loading={loading.backtest}
          result={results.backtest}
          icon={BarChart3}
          accent="purple"
        />

        <ActionButton
          label="Verifier la sante"
          description="Verifie que le backend est operationnel et que le modele est charge."
          onClick={() => run('health', getHealth)}
          loading={loading.health}
          result={results.health}
          icon={Activity}
          accent="racing"
        />
      </div>

      {/* Workflow Guide */}
      <div className="card-glass p-6 border-gold-400/10">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-gold-400/10 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-gold-400" />
          </div>
          <h3 className="font-display font-semibold text-white">Workflow recommande</h3>
        </div>
        <ol className="space-y-3">
          {[
            { step: 1, text: 'Collecter une periode historique (ex: 3 derniers mois) pour alimenter la base.' },
            { step: 2, text: 'Entrainer le modele sur les donnees collectees.' },
            { step: 3, text: "Collecter aujourd'hui pour recuperer les courses du jour." },
            { step: 4, text: 'Generer les predictions pour les courses a venir.' },
            { step: 5, text: 'Apres les courses, Backtester pour comparer les predictions aux resultats.' },
          ].map(({ step, text }) => (
            <li key={step} className="flex items-start gap-3">
              <span className="w-6 h-6 rounded-lg bg-dark-800 flex items-center justify-center flex-shrink-0 text-xs font-mono font-bold text-gold-400">
                {step}
              </span>
              <span className="text-sm text-dark-300 leading-relaxed">{text}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

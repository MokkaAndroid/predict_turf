import { useState } from 'react'
import { postCollect, postCollectToday, postTrain, postPredict, postBacktest, getHealth, postReset } from '../api'
import { Settings, Download, Calendar, Brain, Target, BarChart3, Activity, Loader2, CheckCircle2, XCircle, BookOpen, Trash2 } from 'lucide-react'

function ActionButton({ label, description, onClick, loading, result, icon: Icon, color = 'racing' }) {
  const colors = {
    racing: 'bg-racing-700 hover:bg-racing-600',
    gold: 'bg-gradient-to-r from-gold-500 to-gold-400 text-white hover:from-gold-400 hover:to-gold-300',
    blue: 'bg-blue-600 hover:bg-blue-500',
    purple: 'bg-purple-600 hover:bg-purple-500',
  }

  return (
    <div className="card p-5 hover:shadow-md transition-all duration-200">
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-xl bg-stone-100 flex items-center justify-center flex-shrink-0">
          <Icon className="w-4.5 h-4.5 text-stone-500" />
        </div>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-stone-800 text-sm">{label}</h3>
          <p className="text-xs text-stone-400 mt-1 leading-relaxed">{description}</p>
        </div>
      </div>
      <button onClick={onClick} disabled={loading}
        className={`mt-4 w-full px-4 py-2.5 rounded-xl text-white text-sm font-semibold transition-all duration-200 flex items-center justify-center gap-2 ${colors[color]} shadow-sm hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0`}>
        {loading ? <><Loader2 className="w-4 h-4 animate-spin" /> En cours...</> : 'Lancer'}
      </button>
      {result && (
        <div className="mt-3 relative">
          <div className={`absolute top-2 right-2 ${result.error ? 'text-red-500' : 'text-green-500'}`}>
            {result.error ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
          </div>
          <pre className="bg-stone-50 border border-stone-200 rounded-xl p-3 text-[11px] text-stone-600 overflow-auto max-h-40 font-mono">
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
    try { const result = await fn(); setResults(prev => ({ ...prev, [key]: result })) }
    catch (err) { setResults(prev => ({ ...prev, [key]: { error: err.message } })) }
    setLoading(prev => ({ ...prev, [key]: false }))
  }

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-stone-100 flex items-center justify-center">
          <Settings className="w-5 h-5 text-stone-500" />
        </div>
        <div>
          <h1 className="text-3xl font-display font-bold text-stone-900">Administration</h1>
          <p className="text-stone-400 text-sm mt-0.5">Gestion du pipeline de donnees et du modele</p>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <ActionButton label="Collecter aujourd'hui" description="Recupere les courses de plat du jour depuis l'API PMU."
          onClick={() => run('today', postCollectToday)} loading={loading.today} result={results.today} icon={Download} color="racing" />

        <div className="card p-5 hover:shadow-md transition-all duration-200">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-xl bg-stone-100 flex items-center justify-center flex-shrink-0">
              <Calendar className="w-4.5 h-4.5 text-stone-500" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-stone-800 text-sm">Collecter une periode</h3>
              <p className="text-xs text-stone-400 mt-1 leading-relaxed">Recupere l'historique des courses sur une plage de dates.</p>
            </div>
          </div>
          <div className="flex gap-2 mt-4">
            <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)}
              className="flex-1 bg-white border border-stone-200 rounded-xl px-3 py-2.5 text-sm text-stone-700 focus:outline-none focus:border-racing-300 focus:ring-2 focus:ring-racing-100" />
            <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)}
              className="flex-1 bg-white border border-stone-200 rounded-xl px-3 py-2.5 text-sm text-stone-700 focus:outline-none focus:border-racing-300 focus:ring-2 focus:ring-racing-100" />
          </div>
          <button onClick={() => run('range', () => postCollect(startDate, endDate))} disabled={loading.range || !startDate || !endDate}
            className="mt-3 w-full px-4 py-2.5 rounded-xl text-white text-sm font-semibold bg-racing-700 hover:bg-racing-600 shadow-sm hover:shadow-md transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 flex items-center justify-center gap-2">
            {loading.range ? <><Loader2 className="w-4 h-4 animate-spin" /> En cours...</> : 'Collecter'}
          </button>
          {results.range && (
            <div className="mt-3 relative">
              <div className={`absolute top-2 right-2 ${results.range.error ? 'text-red-500' : 'text-green-500'}`}>
                {results.range.error ? <XCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
              </div>
              <pre className="bg-stone-50 border border-stone-200 rounded-xl p-3 text-[11px] text-stone-600 overflow-auto max-h-40 font-mono">
                {JSON.stringify(results.range, null, 2)}
              </pre>
            </div>
          )}
        </div>

        <ActionButton label="Entrainer le modele" description="Entraine le modele ML (LightGBM) sur toutes les courses terminees."
          onClick={() => run('train', postTrain)} loading={loading.train} result={results.train} icon={Brain} color="blue" />

        <ActionButton label="Generer les predictions" description="Genere les predictions pour toutes les courses a venir."
          onClick={() => run('predict', postPredict)} loading={loading.predict} result={results.predict} icon={Target} color="gold" />

        <ActionButton label="Backtester" description="Compare les predictions aux resultats reels et calcule les gains."
          onClick={() => run('backtest', postBacktest)} loading={loading.backtest} result={results.backtest} icon={BarChart3} color="purple" />

        <ActionButton label="Verifier la sante" description="Verifie que le backend est operationnel et que le modele est charge."
          onClick={() => run('health', getHealth)} loading={loading.health} result={results.health} icon={Activity} color="racing" />

        <div className="card p-5 hover:shadow-md hover:border-red-200 transition-all duration-200">
          <div className="flex items-start gap-3">
            <div className="w-9 h-9 rounded-xl bg-red-50 flex items-center justify-center flex-shrink-0">
              <Trash2 className="w-4.5 h-4.5 text-red-400" />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="font-semibold text-stone-800 text-sm">Reinitialiser la base</h3>
              <p className="text-xs text-stone-400 mt-1 leading-relaxed">Supprime toutes les donnees. Action irreversible.</p>
            </div>
          </div>
          <button onClick={() => { if (window.confirm('Supprimer TOUTES les donnees ?')) run('reset', postReset) }} disabled={loading.reset}
            className="mt-4 w-full px-4 py-2.5 rounded-xl text-white text-sm font-semibold bg-red-500 hover:bg-red-400 shadow-sm hover:shadow-md transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed hover:-translate-y-0.5 active:translate-y-0 flex items-center justify-center gap-2">
            {loading.reset ? <><Loader2 className="w-4 h-4 animate-spin" /> En cours...</> : 'Reinitialiser'}
          </button>
          {results.reset && (
            <pre className="mt-3 bg-stone-50 border border-stone-200 rounded-xl p-3 text-[11px] text-stone-600 overflow-auto max-h-40 font-mono">
              {JSON.stringify(results.reset, null, 2)}
            </pre>
          )}
        </div>
      </div>

      <div className="card p-6 border-gold-200/50 bg-gold-50/30">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 rounded-lg bg-gold-100 flex items-center justify-center">
            <BookOpen className="w-4 h-4 text-gold-600" />
          </div>
          <h3 className="font-display font-semibold text-stone-800">Workflow recommande</h3>
        </div>
        <ol className="space-y-3">
          {[
            'Collecter une periode historique (ex: 3 derniers mois) pour alimenter la base.',
            'Entrainer le modele sur les donnees collectees.',
            "Collecter aujourd'hui pour recuperer les courses du jour.",
            'Generer les predictions pour les courses a venir.',
            'Apres les courses, Backtester pour comparer les predictions aux resultats.',
          ].map((text, i) => (
            <li key={i} className="flex items-start gap-3">
              <span className="w-6 h-6 rounded-lg bg-gold-100 flex items-center justify-center flex-shrink-0 text-xs font-mono font-bold text-gold-700">{i + 1}</span>
              <span className="text-sm text-stone-600 leading-relaxed">{text}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

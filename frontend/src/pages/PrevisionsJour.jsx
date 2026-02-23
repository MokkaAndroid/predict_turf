import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { getPrevisionsJour, postDailyUpdate } from '../api'
import { Target, Filter, Zap, Users, MapPin, ChevronRight, RefreshCw } from 'lucide-react'

function ConfidenceGauge({ value }) {
  const pct = Math.min(value, 100)
  let barColor = 'from-red-400 to-red-300'
  let textColor = 'text-red-500'
  if (pct >= 60) {
    barColor = 'from-racing-600 to-racing-500'
    textColor = 'text-racing-600'
  } else if (pct >= 35) {
    barColor = 'from-gold-500 to-gold-400'
    textColor = 'text-gold-600'
  }

  return (
    <div className="flex items-center gap-3">
      <div className="w-28 h-2 bg-stone-200 rounded-full overflow-hidden">
        <div className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-sm font-mono font-bold ${textColor}`}>{value.toFixed(0)}%</span>
    </div>
  )
}

function CoteBadge({ cote }) {
  if (!cote) return <span className="text-stone-300">-</span>
  let style = 'bg-stone-100 text-stone-600 border-stone-200'
  if (cote <= 3) style = 'bg-green-50 text-green-700 border-green-200'
  else if (cote <= 6) style = 'bg-blue-50 text-blue-700 border-blue-200'
  else if (cote <= 10) style = 'bg-gold-50 text-gold-700 border-gold-200'
  else style = 'bg-red-50 text-red-600 border-red-200'

  return (
    <span className={`text-lg font-mono font-bold px-3 py-1.5 rounded-xl border ${style}`}>{cote.toFixed(1)}</span>
  )
}

function RankBadge({ rank }) {
  if (rank === 1) return (
    <div className="w-14 flex items-center justify-center bg-gradient-to-b from-gold-400 to-gold-500 rounded-l-2xl text-white font-display font-bold text-xl shadow-inner">1</div>
  )
  if (rank === 2) return (
    <div className="w-14 flex items-center justify-center bg-gradient-to-b from-stone-300 to-stone-400 rounded-l-2xl text-white font-display font-bold text-xl">2</div>
  )
  if (rank === 3) return (
    <div className="w-14 flex items-center justify-center bg-gradient-to-b from-gold-600 to-gold-700 rounded-l-2xl text-white font-display font-bold text-xl">3</div>
  )
  return (
    <div className="w-14 flex items-center justify-center bg-stone-100 rounded-l-2xl text-stone-500 font-mono font-bold text-lg">{rank}</div>
  )
}

export default function PrevisionsJour() {
  const [previsions, setPrevisions] = useState([])
  const [loading, setLoading] = useState(true)
  const [filterReunion, setFilterReunion] = useState('all')
  const [filterCourse, setFilterCourse] = useState('all')
  const [updating, setUpdating] = useState(false)
  const [updateResult, setUpdateResult] = useState(null)

  const loadPrevisions = () => {
    setLoading(true)
    getPrevisionsJour()
      .then(setPrevisions)
      .catch(() => setPrevisions([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadPrevisions() }, [])

  const handleDailyUpdate = async () => {
    setUpdating(true)
    setUpdateResult(null)
    try {
      const result = await postDailyUpdate()
      setUpdateResult(result)
      // Recharger les prévisions avec les données à jour
      loadPrevisions()
    } catch (e) {
      setUpdateResult({ status: 'error', error: e.message })
    } finally {
      setUpdating(false)
    }
  }

  const reunions = useMemo(() => {
    const map = new Map()
    previsions.forEach(p => {
      const key = `R${p.numero_reunion}`
      if (!map.has(key)) map.set(key, { key, hippodrome: p.hippodrome, numero: p.numero_reunion })
    })
    return [...map.values()].sort((a, b) => a.numero - b.numero)
  }, [previsions])

  const coursesDisponibles = useMemo(() => {
    let source = previsions
    if (filterReunion !== 'all') {
      const numReunion = parseInt(filterReunion.replace('R', ''))
      source = source.filter(p => p.numero_reunion === numReunion)
    }
    const set = new Set()
    return source
      .filter(p => { const key = `R${p.numero_reunion}C${p.numero_course}`; if (set.has(key)) return false; set.add(key); return true })
      .map(p => ({ key: `R${p.numero_reunion}C${p.numero_course}`, label: `C${p.numero_course} - ${p.heure} (${p.distance}m)`, numero_reunion: p.numero_reunion, numero_course: p.numero_course }))
      .sort((a, b) => a.numero_course - b.numero_course)
  }, [previsions, filterReunion])

  useEffect(() => { setFilterCourse('all') }, [filterReunion])

  const filtered = useMemo(() => {
    let result = previsions
    if (filterReunion !== 'all') {
      const numReunion = parseInt(filterReunion.replace('R', ''))
      result = result.filter(p => p.numero_reunion === numReunion)
    }
    if (filterCourse !== 'all') {
      const match = filterCourse.match(/R(\d+)C(\d+)/)
      if (match) { result = result.filter(p => p.numero_reunion === parseInt(match[1]) && p.numero_course === parseInt(match[2])) }
    }
    return result
  }, [previsions, filterReunion, filterCourse])

  if (loading) return (
    <div className="loading-pulse">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-racing-200 border-t-racing-600 rounded-full animate-spin" />
        <span className="text-sm text-stone-400">Chargement des previsions...</span>
      </div>
    </div>
  )

  const now = new Date()
  const dateStr = now.toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })

  return (
    <div className="space-y-6 animate-slide-up">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gold-100 flex items-center justify-center">
            <Target className="w-5 h-5 text-gold-600" />
          </div>
          <div>
            <h1 className="text-3xl font-display font-bold text-stone-900">Previsions du jour</h1>
            <p className="text-stone-400 mt-0.5 capitalize text-sm">{dateStr}</p>
          </div>
        </div>
        <button
          onClick={handleDailyUpdate}
          disabled={updating}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-xl font-semibold text-sm transition-all shadow-sm ${
            updating
              ? 'bg-stone-100 text-stone-400 cursor-wait'
              : 'bg-racing-700 text-white hover:bg-racing-800 hover:shadow-md active:scale-95'
          }`}
        >
          <RefreshCw className={`w-4 h-4 ${updating ? 'animate-spin' : ''}`} />
          {updating ? 'Mise a jour...' : 'Mise a jour quotidienne'}
        </button>
      </div>

      {/* Feedback mise à jour */}
      {updateResult && (
        <div className={`card p-4 text-sm ${updateResult.status === 'ok' ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}>
          {updateResult.status === 'ok' ? (
            <div className="flex items-center gap-4 text-green-700">
              <span className="font-semibold">Mise a jour terminee</span>
              <span>{updateResult.courses_mises_a_jour} course(s) rafraichie(s)</span>
              <span>{updateResult.courses_creees} nouvelle(s)</span>
              <span>{updateResult.predictions_generees} prediction(s) generee(s)</span>
              {!updateResult.model_loaded && <span className="text-gold-600 font-medium">Modele non charge — entrainez-le depuis l'admin</span>}
            </div>
          ) : (
            <span className="text-red-600">Erreur : {updateResult.error}</span>
          )}
        </div>
      )}

      {previsions.length === 0 ? (
        <div className="card p-8 text-center border-gold-200 bg-gold-50/50">
          <Zap className="w-8 h-8 text-gold-500 mx-auto mb-3" />
          <p className="text-gold-700 font-medium">Aucune prevision pour aujourd'hui</p>
          <p className="text-stone-400 text-sm mt-1">Utilisez l'onglet Administration pour collecter les courses du jour et generer les predictions.</p>
        </div>
      ) : (
        <>
          {/* Filters */}
          <div className="card p-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2 text-stone-400">
                <Filter className="w-4 h-4" />
                <span className="text-xs font-semibold uppercase tracking-wider">Filtrer</span>
              </div>
              <div className="flex items-center gap-2">
                <label className="text-[10px] text-stone-400 uppercase tracking-wider">Reunion</label>
                <div className="flex flex-wrap gap-1">
                  <button onClick={() => setFilterReunion('all')} className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${filterReunion === 'all' ? 'bg-racing-700 text-white shadow-sm' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}>Toutes</button>
                  {reunions.map(r => (
                    <button key={r.key} onClick={() => setFilterReunion(r.key)} className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${filterReunion === r.key ? 'bg-racing-700 text-white shadow-sm' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}>
                      <span className="font-bold">{r.key}</span> <span className="opacity-60">{r.hippodrome}</span>
                    </button>
                  ))}
                </div>
              </div>
              {filterReunion !== 'all' && coursesDisponibles.length > 1 && (
                <div className="flex items-center gap-2">
                  <label className="text-[10px] text-stone-400 uppercase tracking-wider">Course</label>
                  <div className="flex flex-wrap gap-1">
                    <button onClick={() => setFilterCourse('all')} className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${filterCourse === 'all' ? 'bg-gold-500 text-white shadow-sm' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}>Toutes</button>
                    {coursesDisponibles.map(c => (
                      <button key={c.key} onClick={() => setFilterCourse(c.key)} className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${filterCourse === c.key ? 'bg-gold-500 text-white shadow-sm' : 'bg-stone-100 text-stone-600 hover:bg-stone-200'}`}>{c.label}</button>
                    ))}
                  </div>
                </div>
              )}
              <span className="ml-auto text-xs text-stone-400 font-mono">{filtered.length} / {previsions.length}</span>
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-6 text-[11px] text-stone-400">
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-racing-500" /> Forte</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-gold-500" /> Moyenne</span>
            <span className="flex items-center gap-1.5"><span className="w-2.5 h-2.5 rounded-full bg-red-400" /> Faible</span>
            <span className="flex items-center gap-1.5"><span className="px-1.5 py-0.5 rounded bg-purple-100 text-purple-600 text-[10px] font-bold">VB</span> Value Bet</span>
          </div>

          {/* Prediction Cards */}
          {filtered.length === 0 ? (
            <div className="text-center py-12 text-stone-400">Aucune prevision pour ce filtre</div>
          ) : (
            <div className="space-y-3">
              {filtered.map((p, idx) => (
                <Link key={p.course_id} to={`/course/${p.course_id}`} className="block card-hover group">
                  <div className="flex items-stretch">
                    <RankBadge rank={idx + 1} />
                    <div className="flex-1 p-5">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <span className="text-xl font-display font-bold text-stone-800 group-hover:text-racing-700 transition-colors">{p.cheval_nom}</span>
                            <span className="text-xs text-stone-400 bg-stone-100 px-2 py-0.5 rounded-lg font-mono">N{p.numero}</span>
                            {p.top5_confiance && (
                              <span className="text-[10px] font-bold px-2 py-1 rounded-lg bg-blue-100 text-blue-700 border border-blue-200 uppercase tracking-wider">Top 5</span>
                            )}
                            {p.is_value_bet && (
                              <span className="text-[10px] font-bold px-2 py-1 rounded-lg bg-purple-100 text-purple-600 border border-purple-200 uppercase tracking-wider">Value Bet</span>
                            )}
                          </div>
                          <div className="flex items-center gap-4 mt-2 text-xs text-stone-400">
                            <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> <span className="text-stone-600 font-medium">{p.hippodrome}</span></span>
                            <span className="font-mono">R{p.numero_reunion}C{p.numero_course}</span>
                            <span className="font-mono">{p.heure}</span>
                            <span>{p.distance}m</span>
                            <span className="flex items-center gap-1"><Users className="w-3 h-3" /> {p.nombre_partants}</span>
                          </div>
                          <div className="flex items-center gap-4 mt-1 text-xs text-stone-400">
                            {p.jockey_nom && <span>Jockey: <span className="text-stone-600">{p.jockey_nom}</span></span>}
                            {p.entraineur_nom && <span>Entr: <span className="text-stone-600">{p.entraineur_nom}</span></span>}
                          </div>
                        </div>
                        <div className="flex items-start gap-6 ml-4">
                          <div className="text-center">
                            <span className="text-[10px] uppercase tracking-wider text-stone-400 block mb-1">Cote</span>
                            <CoteBadge cote={p.cote} />
                          </div>
                          <div className="text-center">
                            <span className="text-[10px] uppercase tracking-wider text-stone-400 block mb-1">Proba</span>
                            <span className="text-2xl font-mono font-bold text-racing-600">{(p.probabilite * 100).toFixed(0)}%</span>
                          </div>
                        </div>
                      </div>
                      <div className="mt-4 flex items-center gap-4">
                        <span className="text-[10px] text-stone-400 uppercase tracking-wider w-16">Confiance</span>
                        <ConfidenceGauge value={p.score_confiance} />
                      </div>
                      {p.commentaire && (
                        <p className="mt-3 text-xs text-stone-500 italic bg-stone-50 rounded-xl px-4 py-2.5 border border-stone-100">{p.commentaire}</p>
                      )}
                    </div>
                    <div className="flex items-center pr-4 text-stone-300 group-hover:text-stone-500 transition-colors">
                      <ChevronRight className="w-5 h-5" />
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          )}

          {/* Summary Table */}
          <div className="card p-6">
            <h2 className="font-display font-semibold text-stone-800 text-lg mb-4">Resume rapide</h2>
            <div className="overflow-x-auto">
              <table className="w-full table-premium">
                <thead>
                  <tr><th>#</th><th>Heure</th><th>Hippodrome</th><th>Course</th><th>N</th><th>Cheval</th><th className="text-right">Cote</th><th className="text-right">Confiance</th><th className="text-center">Top 5</th><th className="text-center">VB</th></tr>
                </thead>
                <tbody>
                  {filtered.map((p, idx) => (
                    <tr key={p.course_id} className={idx < 3 ? '!bg-racing-50/50' : ''}>
                      <td className="font-display font-bold text-stone-600">{idx + 1}</td>
                      <td className="text-stone-400 font-mono text-xs">{p.heure}</td>
                      <td className="font-medium text-stone-700">{p.hippodrome}</td>
                      <td className="text-stone-400 font-mono text-xs">R{p.numero_reunion}C{p.numero_course}</td>
                      <td className="font-mono font-bold text-stone-700">{p.numero}</td>
                      <td className="font-semibold text-stone-800">{p.cheval_nom}</td>
                      <td className={`text-right font-mono font-bold ${p.cote <= 3 ? 'text-green-600' : p.cote <= 6 ? 'text-blue-600' : p.cote <= 10 ? 'text-gold-600' : 'text-red-500'}`}>{p.cote?.toFixed(1) || '-'}</td>
                      <td className={`text-right font-mono font-bold ${p.score_confiance >= 60 ? 'text-racing-600' : p.score_confiance >= 35 ? 'text-gold-600' : 'text-red-500'}`}>{p.score_confiance.toFixed(0)}%</td>
                      <td className="text-center">{p.top5_confiance ? <span className="text-[10px] font-bold text-blue-700 bg-blue-100 px-2 py-0.5 rounded-lg">T5</span> : <span className="text-stone-300">-</span>}</td>
                      <td className="text-center">{p.is_value_bet ? <span className="text-[10px] font-bold text-purple-600 bg-purple-100 px-2 py-0.5 rounded-lg">VB</span> : <span className="text-stone-300">-</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

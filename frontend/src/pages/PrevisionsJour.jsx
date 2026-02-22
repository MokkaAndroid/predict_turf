import { useState, useEffect, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { getPrevisionsJour } from '../api'
import { Target, Filter, Star, Zap, Users, MapPin, ChevronRight } from 'lucide-react'

function ConfidenceGauge({ value }) {
  const pct = Math.min(value, 100)
  let barColor = 'from-red-500 to-red-400'
  let textColor = 'text-red-400'
  if (pct >= 60) {
    barColor = 'from-racing-500 to-racing-400'
    textColor = 'text-racing-400'
  } else if (pct >= 35) {
    barColor = 'from-gold-500 to-gold-400'
    textColor = 'text-gold-400'
  }

  return (
    <div className="flex items-center gap-3">
      <div className="w-28 h-2 bg-dark-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barColor} transition-all duration-500`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-sm font-mono font-bold ${textColor}`}>
        {value.toFixed(0)}%
      </span>
    </div>
  )
}

function CoteBadge({ cote }) {
  if (!cote) return <span className="text-dark-600">-</span>
  let style = 'bg-dark-800 text-dark-300 border-dark-700'
  if (cote <= 3) style = 'bg-racing-800/50 text-racing-300 border-racing-700/50'
  else if (cote <= 6) style = 'bg-blue-900/30 text-blue-300 border-blue-800/30'
  else if (cote <= 10) style = 'bg-gold-400/10 text-gold-300 border-gold-400/20'
  else style = 'bg-red-900/30 text-red-300 border-red-800/30'

  return (
    <span className={`text-lg font-mono font-bold px-3 py-1.5 rounded-xl border ${style}`}>
      {cote.toFixed(1)}
    </span>
  )
}

function RankBadge({ rank }) {
  if (rank === 1) return (
    <div className="w-14 flex items-center justify-center bg-gradient-to-b from-gold-400 to-gold-500 rounded-l-2xl text-racing-950 font-display font-bold text-xl">
      1
    </div>
  )
  if (rank === 2) return (
    <div className="w-14 flex items-center justify-center bg-gradient-to-b from-dark-300 to-dark-400 rounded-l-2xl text-dark-900 font-display font-bold text-xl">
      2
    </div>
  )
  if (rank === 3) return (
    <div className="w-14 flex items-center justify-center bg-gradient-to-b from-gold-700 to-gold-800 rounded-l-2xl text-gold-200 font-display font-bold text-xl">
      3
    </div>
  )
  return (
    <div className="w-14 flex items-center justify-center bg-dark-800 rounded-l-2xl text-dark-400 font-mono font-bold text-lg">
      {rank}
    </div>
  )
}

export default function PrevisionsJour() {
  const [previsions, setPrevisions] = useState([])
  const [loading, setLoading] = useState(true)
  const [filterReunion, setFilterReunion] = useState('all')
  const [filterCourse, setFilterCourse] = useState('all')

  useEffect(() => {
    getPrevisionsJour()
      .then(setPrevisions)
      .catch(() => setPrevisions([]))
      .finally(() => setLoading(false))
  }, [])

  const reunions = useMemo(() => {
    const map = new Map()
    previsions.forEach(p => {
      const key = `R${p.numero_reunion}`
      if (!map.has(key)) {
        map.set(key, { key, label: `R${p.numero_reunion} - ${p.hippodrome}`, hippodrome: p.hippodrome, numero: p.numero_reunion })
      }
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
      .filter(p => {
        const key = `R${p.numero_reunion}C${p.numero_course}`
        if (set.has(key)) return false
        set.add(key)
        return true
      })
      .map(p => ({
        key: `R${p.numero_reunion}C${p.numero_course}`,
        label: `C${p.numero_course} - ${p.heure} (${p.distance}m)`,
        numero_reunion: p.numero_reunion,
        numero_course: p.numero_course,
      }))
      .sort((a, b) => a.numero_course - b.numero_course)
  }, [previsions, filterReunion])

  useEffect(() => {
    setFilterCourse('all')
  }, [filterReunion])

  const filtered = useMemo(() => {
    let result = previsions
    if (filterReunion !== 'all') {
      const numReunion = parseInt(filterReunion.replace('R', ''))
      result = result.filter(p => p.numero_reunion === numReunion)
    }
    if (filterCourse !== 'all') {
      const match = filterCourse.match(/R(\d+)C(\d+)/)
      if (match) {
        const nr = parseInt(match[1])
        const nc = parseInt(match[2])
        result = result.filter(p => p.numero_reunion === nr && p.numero_course === nc)
      }
    }
    return result
  }, [previsions, filterReunion, filterCourse])

  if (loading) {
    return (
      <div className="loading-pulse">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-gold-400/30 border-t-gold-400 rounded-full animate-spin" />
          <span className="text-sm text-dark-400">Chargement des previsions...</span>
        </div>
      </div>
    )
  }

  const now = new Date()
  const dateStr = now.toLocaleDateString('fr-FR', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' })

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gold-400/10 flex items-center justify-center">
            <Target className="w-5 h-5 text-gold-400" />
          </div>
          <div>
            <h1 className="text-3xl font-display font-bold text-white">Previsions du jour</h1>
            <p className="text-dark-400 mt-0.5 capitalize text-sm">{dateStr}</p>
          </div>
        </div>
      </div>

      {previsions.length === 0 ? (
        <div className="card-glass p-8 text-center border-gold-400/20">
          <Zap className="w-8 h-8 text-gold-400 mx-auto mb-3" />
          <p className="text-gold-300 font-medium">Aucune prevision pour aujourd'hui</p>
          <p className="text-dark-400 text-sm mt-1">
            Utilisez l'onglet Administration pour collecter les courses du jour et generer les predictions.
          </p>
        </div>
      ) : (
        <>
          {/* Filters */}
          <div className="card-glass p-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-2 text-dark-400">
                <Filter className="w-4 h-4" />
                <span className="text-xs font-semibold uppercase tracking-wider">Filtrer</span>
              </div>

              <div className="flex items-center gap-2">
                <label className="text-[10px] text-dark-500 uppercase tracking-wider">Reunion</label>
                <div className="flex flex-wrap gap-1">
                  <button
                    onClick={() => setFilterReunion('all')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      filterReunion === 'all'
                        ? 'bg-racing-600 text-white shadow-md'
                        : 'bg-dark-800 text-dark-300 hover:bg-dark-700 hover:text-white'
                    }`}
                  >
                    Toutes
                  </button>
                  {reunions.map(r => (
                    <button
                      key={r.key}
                      onClick={() => setFilterReunion(r.key)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                        filterReunion === r.key
                          ? 'bg-racing-600 text-white shadow-md'
                          : 'bg-dark-800 text-dark-300 hover:bg-dark-700 hover:text-white'
                      }`}
                    >
                      <span className="font-bold">{r.key}</span>
                      <span className="ml-1 opacity-60">{r.hippodrome}</span>
                    </button>
                  ))}
                </div>
              </div>

              {filterReunion !== 'all' && coursesDisponibles.length > 1 && (
                <div className="flex items-center gap-2">
                  <label className="text-[10px] text-dark-500 uppercase tracking-wider">Course</label>
                  <div className="flex flex-wrap gap-1">
                    <button
                      onClick={() => setFilterCourse('all')}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                        filterCourse === 'all'
                          ? 'bg-gold-500 text-racing-950 shadow-gold'
                          : 'bg-dark-800 text-dark-300 hover:bg-dark-700 hover:text-white'
                      }`}
                    >
                      Toutes
                    </button>
                    {coursesDisponibles.map(c => (
                      <button
                        key={c.key}
                        onClick={() => setFilterCourse(c.key)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                          filterCourse === c.key
                            ? 'bg-gold-500 text-racing-950 shadow-gold'
                            : 'bg-dark-800 text-dark-300 hover:bg-dark-700 hover:text-white'
                        }`}
                      >
                        {c.label}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              <span className="ml-auto text-xs text-dark-500 font-mono">
                {filtered.length} / {previsions.length}
              </span>
            </div>
          </div>

          {/* Legend */}
          <div className="flex items-center gap-6 text-[11px] text-dark-400">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-racing-400" /> Forte (&ge;60%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-gold-400" /> Moyenne (35-59%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-red-400" /> Faible (&lt;35%)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300 text-[10px] font-bold">VB</span> Value Bet
            </span>
          </div>

          {/* Prediction Cards */}
          {filtered.length === 0 ? (
            <div className="text-center py-12 text-dark-500">Aucune prevision pour ce filtre</div>
          ) : (
            <div className="space-y-3">
              {filtered.map((p, idx) => (
                <Link
                  key={p.course_id}
                  to={`/course/${p.course_id}`}
                  className="block card-glass hover:border-dark-600/80 hover:shadow-premium-lg transition-all duration-300 group"
                >
                  <div className="flex items-stretch">
                    <RankBadge rank={idx + 1} />

                    <div className="flex-1 p-5">
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <div className="flex items-center gap-3">
                            <span className="text-xl font-display font-bold text-white group-hover:text-gold-300 transition-colors">
                              {p.cheval_nom}
                            </span>
                            <span className="text-xs text-dark-400 bg-dark-800 px-2 py-0.5 rounded-lg font-mono">
                              N{p.numero}
                            </span>
                            {p.is_value_bet && (
                              <span className="text-[10px] font-bold px-2 py-1 rounded-lg bg-purple-500/20 text-purple-300 border border-purple-500/30 uppercase tracking-wider">
                                Value Bet
                              </span>
                            )}
                          </div>
                          <div className="flex items-center gap-4 mt-2 text-xs text-dark-400">
                            <span className="flex items-center gap-1">
                              <MapPin className="w-3 h-3" />
                              <span className="text-dark-200 font-medium">{p.hippodrome}</span>
                            </span>
                            <span className="font-mono">R{p.numero_reunion}C{p.numero_course}</span>
                            <span className="font-mono">{p.heure}</span>
                            <span>{p.distance}m</span>
                            <span className="flex items-center gap-1">
                              <Users className="w-3 h-3" />
                              {p.nombre_partants}
                            </span>
                          </div>
                          <div className="flex items-center gap-4 mt-1 text-xs text-dark-500">
                            {p.jockey_nom && <span>Jockey: <span className="text-dark-300">{p.jockey_nom}</span></span>}
                            {p.entraineur_nom && <span>Entr: <span className="text-dark-300">{p.entraineur_nom}</span></span>}
                          </div>
                        </div>

                        <div className="flex items-start gap-6 ml-4">
                          <div className="text-center">
                            <span className="text-[10px] uppercase tracking-wider text-dark-500 block mb-1">Cote</span>
                            <CoteBadge cote={p.cote} />
                          </div>
                          <div className="text-center">
                            <span className="text-[10px] uppercase tracking-wider text-dark-500 block mb-1">Proba</span>
                            <span className="text-2xl font-mono font-bold text-racing-400">
                              {(p.probabilite * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="mt-4 flex items-center gap-4">
                        <span className="text-[10px] text-dark-500 uppercase tracking-wider w-16">Confiance</span>
                        <ConfidenceGauge value={p.score_confiance} />
                      </div>

                      {p.commentaire && (
                        <p className="mt-3 text-xs text-dark-400 italic bg-dark-800/50 rounded-xl px-4 py-2.5 border border-dark-700/30">
                          {p.commentaire}
                        </p>
                      )}
                    </div>

                    <div className="flex items-center pr-4 text-dark-600 group-hover:text-dark-400 transition-colors">
                      <ChevronRight className="w-5 h-5" />
                    </div>
                  </div>
                </Link>
              ))}
            </div>
          )}

          {/* Summary Table */}
          <div className="card-glass p-6">
            <h2 className="font-display font-semibold text-white text-lg mb-4">Resume rapide</h2>
            <div className="overflow-x-auto">
              <table className="w-full table-premium">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Heure</th>
                    <th>Hippodrome</th>
                    <th>Course</th>
                    <th>N</th>
                    <th>Cheval</th>
                    <th className="text-right">Cote</th>
                    <th className="text-right">Confiance</th>
                    <th className="text-center">VB</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((p, idx) => (
                    <tr key={p.course_id} className={idx < 3 ? '!bg-racing-900/20' : ''}>
                      <td className="font-display font-bold text-dark-200">{idx + 1}</td>
                      <td className="text-dark-400 font-mono text-xs">{p.heure}</td>
                      <td className="font-medium text-dark-200">{p.hippodrome}</td>
                      <td className="text-dark-400 font-mono text-xs">R{p.numero_reunion}C{p.numero_course}</td>
                      <td className="font-mono font-bold text-dark-200">{p.numero}</td>
                      <td className="font-semibold text-white">{p.cheval_nom}</td>
                      <td className={`text-right font-mono font-bold ${
                        p.cote <= 3 ? 'text-racing-400' : p.cote <= 6 ? 'text-blue-400' : p.cote <= 10 ? 'text-gold-400' : 'text-red-400'
                      }`}>
                        {p.cote?.toFixed(1) || '-'}
                      </td>
                      <td className={`text-right font-mono font-bold ${
                        p.score_confiance >= 60 ? 'text-racing-400' : p.score_confiance >= 35 ? 'text-gold-400' : 'text-red-400'
                      }`}>
                        {p.score_confiance.toFixed(0)}%
                      </td>
                      <td className="text-center">
                        {p.is_value_bet ? (
                          <span className="text-[10px] font-bold text-purple-300 bg-purple-500/20 px-2 py-0.5 rounded-lg">VB</span>
                        ) : <span className="text-dark-600">-</span>}
                      </td>
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

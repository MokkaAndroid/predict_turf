import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { getBacktestingStats, getCoursesPassees, getCoursesAVenir, getPrevisionsJour } from '../api'
import { TrendingUp, TrendingDown, BarChart3, Target, Trophy, ArrowRight, Clock, ChevronRight, Zap, CalendarDays } from 'lucide-react'

function StatCard({ label, value, sub, icon: Icon, positive }) {
  return (
    <div className="card p-5 group hover:shadow-md transition-all duration-200">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs font-semibold text-stone-400 uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-bold text-stone-900 mt-2 font-mono">{value}</p>
          {sub && <p className="text-xs text-stone-400 mt-1">{sub}</p>}
        </div>
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
          positive ? 'bg-racing-50 text-racing-600' : 'bg-stone-100 text-stone-400'
        }`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  )
}

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div className="bg-white border border-stone-200 rounded-xl px-4 py-3 shadow-lg">
      <p className="text-xs text-stone-400 mb-1">Course #{label}</p>
      {payload.map((p, i) => (
        <p key={i} className="text-sm font-mono font-semibold" style={{ color: p.color }}>
          {p.name}: {p.value > 0 ? '+' : ''}{p.value} EUR
        </p>
      ))}
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [passees, setPassees] = useState([])
  const [aVenir, setAVenir] = useState([])
  const [top5, setTop5] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([
      getBacktestingStats().catch(() => null),
      getCoursesPassees(20).catch(() => []),
      getCoursesAVenir(10).catch(() => []),
      getPrevisionsJour().catch(() => []),
    ]).then(([s, p, a, t]) => {
      setStats(s)
      setPassees(p)
      setAVenir(a)
      setTop5(t.slice(0, 5))
      setLoading(false)
    })
  }, [])

  if (loading) {
    return (
      <div className="loading-pulse">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-racing-200 border-t-racing-600 rounded-full animate-spin" />
          <span className="text-sm text-stone-400">Chargement des donnees...</span>
        </div>
      </div>
    )
  }

  let cumGagnant = 0
  let cumPlace = 0
  const roiData = passees
    .filter(c => c.gain_simule_gagnant != null)
    .reverse()
    .map((c, i) => {
      cumGagnant += c.gain_simule_gagnant || 0
      cumPlace += c.gain_simule_place || 0
      return {
        course: i + 1,
        gagnant: +cumGagnant.toFixed(2),
        place: +cumPlace.toFixed(2),
      }
    })

  return (
    <div className="space-y-8 animate-slide-up">
      <div>
        <h1 className="text-3xl font-display font-bold text-stone-900">Dashboard</h1>
        <p className="text-stone-400 mt-1">Vue d'ensemble de vos performances</p>
      </div>

      {stats ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard label="Courses analysees" value={stats.total_courses} icon={BarChart3} positive />
          <StatCard
            label="Taux gagnant"
            value={`${stats.taux_gagnant}%`}
            sub={`${stats.gagnant_correct} / ${stats.courses_predites}`}
            icon={Target}
            positive={stats.taux_gagnant > 15}
          />
          <StatCard
            label="Taux place"
            value={`${stats.taux_place}%`}
            sub={`${stats.place_correct} / ${stats.courses_predites}`}
            icon={Trophy}
            positive={stats.taux_place > 40}
          />
          <StatCard
            label="ROI Gagnant"
            value={`${stats.roi_gagnant > 0 ? '+' : ''}${stats.roi_gagnant}%`}
            sub={`P&L: ${stats.profit_gagnant > 0 ? '+' : ''}${stats.profit_gagnant} EUR`}
            icon={stats.roi_gagnant > 0 ? TrendingUp : TrendingDown}
            positive={stats.roi_gagnant > 0}
          />
        </div>
      ) : (
        <div className="card p-6 border-gold-300/50 bg-gold-50/50">
          <div className="flex items-center gap-3 text-gold-700">
            <Zap className="w-5 h-5" />
            <p className="text-sm">
              Aucune donnee de backtesting disponible. Utilisez l'onglet Administration pour demarrer.
            </p>
          </div>
        </div>
      )}

      {top5.length > 0 && (
        <div className="card p-6">
          <div className="flex justify-between items-center mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gold-100 flex items-center justify-center">
                <Zap className="w-4 h-4 text-gold-600" />
              </div>
              <h2 className="font-display font-semibold text-stone-800 text-lg">Top 5 du jour</h2>
            </div>
            <Link to="/jour" className="flex items-center gap-1 text-sm text-racing-600 hover:text-racing-700 font-medium transition-colors">
              Voir tout <ArrowRight className="w-4 h-4" />
            </Link>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full table-premium">
              <thead>
                <tr>
                  <th>Course</th>
                  <th>Heure</th>
                  <th>N.</th>
                  <th>Cheval</th>
                  <th className="text-right">Cote</th>
                  <th className="text-right">Confiance</th>
                </tr>
              </thead>
              <tbody>
                {top5.map((p, i) => (
                  <tr key={i}>
                    <td>
                      <Link to={`/course/${p.course_id}`} className="text-racing-600 hover:text-racing-700 font-medium">
                        R{p.numero_reunion}C{p.numero_course}
                      </Link>
                    </td>
                    <td className="text-stone-400 font-mono text-xs">{p.heure}</td>
                    <td className="font-mono font-bold text-stone-700">{p.numero}</td>
                    <td className="font-semibold text-stone-800">{p.cheval_nom}</td>
                    <td className="text-right">
                      <span className="inline-block bg-gold-100 text-gold-700 px-2.5 py-1 rounded-lg text-xs font-mono font-bold">
                        {p.cote?.toFixed(1) || '-'}
                      </span>
                    </td>
                    <td className="text-right">
                      <span className={`font-mono font-bold text-sm ${
                        p.score_confiance >= 50 ? 'text-racing-600' : p.score_confiance >= 30 ? 'text-gold-600' : 'text-stone-400'
                      }`}>
                        {p.score_confiance}%
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {roiData.length > 0 && (
        <div className="card p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-8 h-8 rounded-lg bg-racing-50 flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-racing-600" />
            </div>
            <h2 className="font-display font-semibold text-stone-800 text-lg">ROI cumule</h2>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={roiData}>
              <defs>
                <linearGradient id="gradGagnant" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#1e6b3a" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#1e6b3a" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradPlace" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#d4af37" stopOpacity={0.15} />
                  <stop offset="100%" stopColor="#d4af37" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#e7e5e4" />
              <XAxis dataKey="course" stroke="#a8a29e" tick={{ fontSize: 11 }} />
              <YAxis stroke="#a8a29e" tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="gagnant" stroke="#1e6b3a" fill="url(#gradGagnant)" name="Gagnant" strokeWidth={2} />
              <Area type="monotone" dataKey="place" stroke="#d4af37" fill="url(#gradPlace)" name="Place" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-6">
        <div className="card p-6">
          <div className="flex justify-between items-center mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-stone-100 flex items-center justify-center">
                <Clock className="w-4 h-4 text-stone-500" />
              </div>
              <h2 className="font-display font-semibold text-stone-800 text-lg">Dernieres courses</h2>
            </div>
            <Link to="/passees" className="flex items-center gap-1 text-sm text-stone-400 hover:text-stone-600 transition-colors">
              Voir tout <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          {passees.length === 0 ? (
            <p className="text-stone-400 text-sm py-4">Aucune course passee</p>
          ) : (
            <div className="space-y-1">
              {passees.slice(0, 8).map(c => (
                <Link
                  key={c.id}
                  to={`/course/${c.id}`}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-stone-50 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-semibold text-stone-700 group-hover:text-racing-700 transition-colors">{c.hippodrome}</span>
                    <span className="text-xs text-stone-400 font-mono">R{c.numero_reunion}C{c.numero_course}</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {c.prediction_correcte_gagnant != null && (
                      <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-lg ${
                        c.prediction_correcte_gagnant
                          ? 'bg-green-100 text-green-700'
                          : c.prediction_correcte_place
                            ? 'bg-gold-100 text-gold-700'
                            : 'bg-stone-100 text-stone-500'
                      }`}>
                        {c.prediction_correcte_gagnant ? 'Gagnant' : c.prediction_correcte_place ? 'Place' : 'Perdu'}
                      </span>
                    )}
                    {c.gain_simule_gagnant != null && (
                      <span className={`text-xs font-mono font-bold ${c.gain_simule_gagnant > 0 ? 'text-green-600' : 'text-red-500'}`}>
                        {c.gain_simule_gagnant > 0 ? '+' : ''}{c.gain_simule_gagnant?.toFixed(2)}
                      </span>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>

        <div className="card p-6">
          <div className="flex justify-between items-center mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-stone-100 flex items-center justify-center">
                <CalendarDays className="w-4 h-4 text-stone-500" />
              </div>
              <h2 className="font-display font-semibold text-stone-800 text-lg">Courses a venir</h2>
            </div>
            <Link to="/a-venir" className="flex items-center gap-1 text-sm text-stone-400 hover:text-stone-600 transition-colors">
              Voir tout <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          {aVenir.length === 0 ? (
            <p className="text-stone-400 text-sm py-4">Aucune course a venir</p>
          ) : (
            <div className="space-y-1">
              {aVenir.slice(0, 8).map(c => (
                <Link
                  key={c.id}
                  to={`/course/${c.id}`}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-stone-50 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-semibold text-stone-700 group-hover:text-racing-700 transition-colors">{c.hippodrome}</span>
                    <span className="text-xs text-stone-400 font-mono">
                      {new Date(c.date).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className="text-xs text-stone-400">{c.distance}m</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {c.favori_nom && (
                      <span className="text-xs font-semibold text-racing-600">{c.favori_nom}</span>
                    )}
                    {c.favori_confiance != null && (
                      <span className="text-xs font-mono text-stone-400">{c.favori_confiance}%</span>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

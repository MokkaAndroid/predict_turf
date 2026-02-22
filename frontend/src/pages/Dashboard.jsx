import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from 'recharts'
import { getBacktestingStats, getCoursesPassees, getCoursesAVenir, getPrevisionsJour } from '../api'
import { TrendingUp, TrendingDown, BarChart3, Target, Trophy, ArrowRight, Clock, ChevronRight, Zap, CalendarDays } from 'lucide-react'

function StatCard({ label, value, sub, icon: Icon, trend }) {
  const isPositive = trend === 'up'
  return (
    <div className="card-glass p-5 stat-card group hover:border-dark-600/50 transition-all duration-300">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs font-semibold text-dark-400 uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-bold text-white mt-2 font-mono">{value}</p>
          {sub && <p className="text-xs text-dark-400 mt-1">{sub}</p>}
        </div>
        <div className={`w-10 h-10 rounded-xl flex items-center justify-center ${
          isPositive ? 'bg-racing-800/50 text-racing-400' : 'bg-dark-800/50 text-dark-400'
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
    <div className="bg-dark-800 border border-dark-700 rounded-xl px-4 py-3 shadow-premium-lg">
      <p className="text-xs text-dark-400 mb-1">Course #{label}</p>
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
          <div className="w-8 h-8 border-2 border-gold-400/30 border-t-gold-400 rounded-full animate-spin" />
          <span className="text-sm text-dark-400">Chargement des donnees...</span>
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
      {/* Header */}
      <div>
        <h1 className="text-3xl font-display font-bold text-white">Dashboard</h1>
        <p className="text-dark-400 mt-1">Vue d'ensemble de vos performances</p>
      </div>

      {/* Stats Grid */}
      {stats ? (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <StatCard
            label="Courses analysees"
            value={stats.total_courses}
            icon={BarChart3}
            trend="up"
          />
          <StatCard
            label="Taux gagnant"
            value={`${stats.taux_gagnant}%`}
            sub={`${stats.gagnant_correct} / ${stats.courses_predites}`}
            icon={Target}
            trend={stats.taux_gagnant > 15 ? 'up' : 'down'}
          />
          <StatCard
            label="Taux place"
            value={`${stats.taux_place}%`}
            sub={`${stats.place_correct} / ${stats.courses_predites}`}
            icon={Trophy}
            trend={stats.taux_place > 40 ? 'up' : 'down'}
          />
          <StatCard
            label="ROI Gagnant"
            value={`${stats.roi_gagnant > 0 ? '+' : ''}${stats.roi_gagnant}%`}
            sub={`P&L: ${stats.profit_gagnant > 0 ? '+' : ''}${stats.profit_gagnant} EUR`}
            icon={stats.roi_gagnant > 0 ? TrendingUp : TrendingDown}
            trend={stats.roi_gagnant > 0 ? 'up' : 'down'}
          />
        </div>
      ) : (
        <div className="card-glass p-6 border-gold-400/20">
          <div className="flex items-center gap-3 text-gold-300">
            <Zap className="w-5 h-5" />
            <p className="text-sm">
              Aucune donnee de backtesting disponible. Utilisez l'onglet Administration pour demarrer.
            </p>
          </div>
        </div>
      )}

      {/* Top 5 du jour */}
      {top5.length > 0 && (
        <div className="card-glass p-6">
          <div className="flex justify-between items-center mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gold-400/10 flex items-center justify-center">
                <Zap className="w-4 h-4 text-gold-400" />
              </div>
              <h2 className="font-display font-semibold text-white text-lg">Top 5 du jour</h2>
            </div>
            <Link to="/jour" className="flex items-center gap-1 text-sm text-gold-400 hover:text-gold-300 transition-colors">
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
                      <Link to={`/course/${p.course_id}`} className="text-gold-400 hover:text-gold-300 font-medium transition-colors">
                        R{p.numero_reunion}C{p.numero_course}
                      </Link>
                    </td>
                    <td className="text-dark-400 font-mono text-xs">{p.heure}</td>
                    <td className="font-mono font-bold text-dark-200">{p.numero}</td>
                    <td className="font-semibold text-white">{p.cheval_nom}</td>
                    <td className="text-right">
                      <span className="inline-block bg-gold-400/10 text-gold-300 px-2.5 py-1 rounded-lg text-xs font-mono font-bold">
                        {p.cote?.toFixed(1) || '-'}
                      </span>
                    </td>
                    <td className="text-right">
                      <span className={`font-mono font-bold text-sm ${
                        p.score_confiance >= 50 ? 'text-racing-400' : p.score_confiance >= 30 ? 'text-gold-400' : 'text-dark-400'
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

      {/* ROI Chart */}
      {roiData.length > 0 && (
        <div className="card-glass p-6">
          <div className="flex items-center gap-3 mb-5">
            <div className="w-8 h-8 rounded-lg bg-racing-700/50 flex items-center justify-center">
              <TrendingUp className="w-4 h-4 text-racing-400" />
            </div>
            <h2 className="font-display font-semibold text-white text-lg">ROI cumule</h2>
          </div>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={roiData}>
              <defs>
                <linearGradient id="gradGagnant" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3d8a55" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#3d8a55" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gradPlace" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#d4af37" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#d4af37" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a31" />
              <XAxis dataKey="course" stroke="#636371" tick={{ fontSize: 11 }} />
              <YAxis stroke="#636371" tick={{ fontSize: 11 }} />
              <Tooltip content={<CustomTooltip />} />
              <Area type="monotone" dataKey="gagnant" stroke="#3d8a55" fill="url(#gradGagnant)" name="Gagnant" strokeWidth={2} />
              <Area type="monotone" dataKey="place" stroke="#d4af37" fill="url(#gradPlace)" name="Place" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Two columns */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Dernières courses */}
        <div className="card-glass p-6">
          <div className="flex justify-between items-center mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-dark-800 flex items-center justify-center">
                <Clock className="w-4 h-4 text-dark-300" />
              </div>
              <h2 className="font-display font-semibold text-white text-lg">Dernieres courses</h2>
            </div>
            <Link to="/passees" className="flex items-center gap-1 text-sm text-dark-400 hover:text-white transition-colors">
              Voir tout <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          {passees.length === 0 ? (
            <p className="text-dark-500 text-sm py-4">Aucune course passee</p>
          ) : (
            <div className="space-y-1">
              {passees.slice(0, 8).map(c => (
                <Link
                  key={c.id}
                  to={`/course/${c.id}`}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-dark-800/50 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-semibold text-dark-200 group-hover:text-white transition-colors">{c.hippodrome}</span>
                    <span className="text-xs text-dark-500 font-mono">R{c.numero_reunion}C{c.numero_course}</span>
                    {c.favori_nom && (
                      <span className="text-xs text-dark-400">{c.favori_nom}</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    {c.prediction_correcte_gagnant != null && (
                      <span className={`text-[10px] font-bold uppercase tracking-wider px-2 py-1 rounded-lg ${
                        c.prediction_correcte_gagnant
                          ? 'bg-racing-800/50 text-racing-300'
                          : c.prediction_correcte_place
                            ? 'bg-gold-400/10 text-gold-400'
                            : 'bg-dark-800 text-dark-400'
                      }`}>
                        {c.prediction_correcte_gagnant ? 'Gagnant' : c.prediction_correcte_place ? 'Place' : 'Perdu'}
                      </span>
                    )}
                    {c.gain_simule_gagnant != null && (
                      <span className={`text-xs font-mono font-bold ${c.gain_simule_gagnant > 0 ? 'text-racing-400' : 'text-red-400'}`}>
                        {c.gain_simule_gagnant > 0 ? '+' : ''}{c.gain_simule_gagnant?.toFixed(2)}
                      </span>
                    )}
                  </div>
                </Link>
              ))}
            </div>
          )}
        </div>

        {/* Courses à venir */}
        <div className="card-glass p-6">
          <div className="flex justify-between items-center mb-5">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-dark-800 flex items-center justify-center">
                <CalendarDays className="w-4 h-4 text-dark-300" />
              </div>
              <h2 className="font-display font-semibold text-white text-lg">Courses a venir</h2>
            </div>
            <Link to="/a-venir" className="flex items-center gap-1 text-sm text-dark-400 hover:text-white transition-colors">
              Voir tout <ChevronRight className="w-4 h-4" />
            </Link>
          </div>
          {aVenir.length === 0 ? (
            <p className="text-dark-500 text-sm py-4">Aucune course a venir</p>
          ) : (
            <div className="space-y-1">
              {aVenir.slice(0, 8).map(c => (
                <Link
                  key={c.id}
                  to={`/course/${c.id}`}
                  className="flex items-center justify-between p-3 rounded-xl hover:bg-dark-800/50 transition-all group"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-semibold text-dark-200 group-hover:text-white transition-colors">{c.hippodrome}</span>
                    <span className="text-xs text-dark-500 font-mono">
                      {new Date(c.date).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                    </span>
                    <span className="text-xs text-dark-500">{c.distance}m</span>
                  </div>
                  <div className="flex items-center gap-3">
                    {c.favori_nom && (
                      <span className="text-xs font-semibold text-racing-400">{c.favori_nom}</span>
                    )}
                    {c.favori_confiance != null && (
                      <span className="text-xs font-mono text-dark-400">{c.favori_confiance}%</span>
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

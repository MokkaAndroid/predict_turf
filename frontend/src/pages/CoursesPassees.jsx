import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getCoursesPassees } from '../api'
import { Clock, Search, TrendingUp, TrendingDown } from 'lucide-react'

export default function CoursesPassees() {
  const [courses, setCourses] = useState([])
  const [loading, setLoading] = useState(true)
  const [filter, setFilter] = useState('')

  useEffect(() => {
    getCoursesPassees(100, 0, filter)
      .then(setCourses)
      .catch(() => setCourses([]))
      .finally(() => setLoading(false))
  }, [filter])

  if (loading) {
    return (
      <div className="loading-pulse">
        <div className="flex flex-col items-center gap-3">
          <div className="w-8 h-8 border-2 border-gold-400/30 border-t-gold-400 rounded-full animate-spin" />
          <span className="text-sm text-dark-400">Chargement...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6 animate-slide-up">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-dark-800 flex items-center justify-center">
            <Clock className="w-5 h-5 text-dark-300" />
          </div>
          <div>
            <h1 className="text-3xl font-display font-bold text-white">Historique</h1>
            <p className="text-dark-400 text-sm mt-0.5">Backtesting des predictions passees</p>
          </div>
        </div>
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-dark-500" />
          <input
            type="text"
            placeholder="Filtrer par hippodrome..."
            value={filter}
            onChange={e => setFilter(e.target.value)}
            className="bg-dark-800 border border-dark-700 rounded-xl pl-10 pr-4 py-2.5 text-sm text-dark-200 placeholder-dark-500 w-72 focus:outline-none focus:border-gold-400/50 focus:ring-1 focus:ring-gold-400/20 transition-colors"
          />
        </div>
      </div>

      <div className="card-glass overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full table-premium">
            <thead>
              <tr>
                <th>Date</th>
                <th>Hippodrome</th>
                <th>Course</th>
                <th>Distance</th>
                <th>Favori predit</th>
                <th className="text-center">Resultat</th>
                <th className="text-right">Gain Gagnant</th>
                <th className="text-right">Gain Place</th>
              </tr>
            </thead>
            <tbody>
              {courses.map(c => (
                <tr key={c.id}>
                  <td>
                    <Link to={`/course/${c.id}`} className="text-gold-400 hover:text-gold-300 font-medium transition-colors">
                      {new Date(c.date).toLocaleDateString('fr-FR')}
                    </Link>
                  </td>
                  <td className="font-semibold text-dark-200">{c.hippodrome}</td>
                  <td className="text-dark-400 font-mono text-xs">R{c.numero_reunion}C{c.numero_course}</td>
                  <td className="text-dark-400">{c.distance}m</td>
                  <td className="text-dark-200">{c.favori_nom || <span className="text-dark-600">-</span>}</td>
                  <td className="text-center">
                    {c.prediction_correcte_gagnant != null ? (
                      <span className={`inline-block px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider ${
                        c.prediction_correcte_gagnant
                          ? 'bg-racing-800/50 text-racing-300'
                          : c.prediction_correcte_place
                            ? 'bg-gold-400/10 text-gold-400'
                            : 'bg-dark-800 text-dark-400'
                      }`}>
                        {c.prediction_correcte_gagnant ? 'Gagnant' : c.prediction_correcte_place ? 'Place' : 'Perdu'}
                      </span>
                    ) : (
                      <span className="text-dark-600">-</span>
                    )}
                  </td>
                  <td className="text-right">
                    {c.gain_simule_gagnant != null ? (
                      <span className={`font-mono font-bold flex items-center justify-end gap-1 ${
                        c.gain_simule_gagnant > 0 ? 'text-racing-400' : 'text-red-400'
                      }`}>
                        {c.gain_simule_gagnant > 0 ? (
                          <TrendingUp className="w-3 h-3" />
                        ) : (
                          <TrendingDown className="w-3 h-3" />
                        )}
                        {c.gain_simule_gagnant > 0 ? '+' : ''}{c.gain_simule_gagnant.toFixed(2)}
                      </span>
                    ) : <span className="text-dark-600">-</span>}
                  </td>
                  <td className="text-right">
                    {c.gain_simule_place != null ? (
                      <span className={`font-mono font-bold ${
                        c.gain_simule_place > 0 ? 'text-racing-400' : 'text-red-400'
                      }`}>
                        {c.gain_simule_place > 0 ? '+' : ''}{c.gain_simule_place.toFixed(2)}
                      </span>
                    ) : <span className="text-dark-600">-</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {courses.length === 0 && (
          <div className="text-center py-12 text-dark-500">Aucune course passee trouvee</div>
        )}
      </div>
    </div>
  )
}

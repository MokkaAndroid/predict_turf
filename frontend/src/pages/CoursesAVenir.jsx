import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getCoursesAVenir } from '../api'
import { CalendarDays, MapPin, Users, ChevronRight, Timer } from 'lucide-react'

function ConfidenceBar({ value }) {
  let barColor = 'from-red-500 to-red-400'
  if (value >= 60) barColor = 'from-racing-500 to-racing-400'
  else if (value >= 35) barColor = 'from-gold-500 to-gold-400'

  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-dark-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${barColor}`}
          style={{ width: `${Math.min(value, 100)}%` }}
        />
      </div>
      <span className="text-xs text-dark-400 font-mono">{value}%</span>
    </div>
  )
}

export default function CoursesAVenir() {
  const [courses, setCourses] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getCoursesAVenir(100)
      .then(setCourses)
      .catch(() => setCourses([]))
      .finally(() => setLoading(false))
  }, [])

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

  const grouped = {}
  courses.forEach(c => {
    const key = new Date(c.date).toLocaleDateString('fr-FR')
    if (!grouped[key]) grouped[key] = []
    grouped[key].push(c)
  })

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-racing-800/50 flex items-center justify-center">
          <CalendarDays className="w-5 h-5 text-racing-400" />
        </div>
        <div>
          <h1 className="text-3xl font-display font-bold text-white">Courses a venir</h1>
          <p className="text-dark-400 text-sm mt-0.5">Prochaines courses avec predictions</p>
        </div>
      </div>

      {courses.length === 0 ? (
        <div className="card-glass p-8 text-center border-gold-400/20">
          <CalendarDays className="w-8 h-8 text-gold-400 mx-auto mb-3" />
          <p className="text-gold-300 font-medium">Aucune course a venir</p>
          <p className="text-dark-400 text-sm mt-1">
            Utilisez l'onglet Administration pour collecter les courses du jour.
          </p>
        </div>
      ) : (
        Object.entries(grouped).map(([dateStr, group]) => (
          <div key={dateStr}>
            <div className="flex items-center gap-3 mb-4">
              <div className="h-px flex-1 bg-dark-800" />
              <h2 className="text-sm font-semibold text-dark-300 uppercase tracking-wider">{dateStr}</h2>
              <div className="h-px flex-1 bg-dark-800" />
            </div>
            <div className="grid gap-3">
              {group.map(c => (
                <Link
                  key={c.id}
                  to={`/course/${c.id}`}
                  className="card-glass p-5 hover:border-dark-600/80 hover:shadow-premium transition-all duration-300 flex items-center justify-between group"
                >
                  <div className="flex items-center gap-5">
                    <div className="text-center min-w-[64px]">
                      <div className="text-xl font-mono font-bold text-gold-400">
                        {new Date(c.date).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                      </div>
                      <div className="text-[10px] text-dark-500 font-mono uppercase tracking-wider mt-0.5">
                        R{c.numero_reunion}C{c.numero_course}
                      </div>
                    </div>
                    <div className="h-10 w-px bg-dark-700" />
                    <div>
                      <div className="flex items-center gap-2">
                        <MapPin className="w-3.5 h-3.5 text-dark-500" />
                        <span className="font-semibold text-white group-hover:text-gold-300 transition-colors">
                          {c.hippodrome}
                        </span>
                      </div>
                      <div className="flex items-center gap-3 mt-1 text-xs text-dark-400">
                        <span>{c.distance}m</span>
                        <span className="flex items-center gap-1">
                          <Users className="w-3 h-3" />
                          {c.nombre_partants} partants
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    {c.favori_nom ? (
                      <>
                        <div className="text-right">
                          <div className="text-sm font-semibold text-racing-400">{c.favori_nom}</div>
                          <div className="text-[10px] text-dark-500 uppercase tracking-wider">Favori modele</div>
                        </div>
                        {c.favori_confiance != null && (
                          <ConfidenceBar value={c.favori_confiance} />
                        )}
                      </>
                    ) : (
                      <span className="text-xs text-dark-500 italic">Pas de prediction</span>
                    )}
                    <ChevronRight className="w-4 h-4 text-dark-600 group-hover:text-dark-400 transition-colors" />
                  </div>
                </Link>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  )
}

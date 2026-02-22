import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { getCoursesAVenir } from '../api'
import { CalendarDays, MapPin, Users, ChevronRight } from 'lucide-react'

function ConfidenceBar({ value }) {
  let barColor = 'from-red-400 to-red-300'
  if (value >= 60) barColor = 'from-racing-600 to-racing-500'
  else if (value >= 35) barColor = 'from-gold-500 to-gold-400'

  return (
    <div className="flex items-center gap-2">
      <div className="w-20 h-1.5 bg-stone-200 rounded-full overflow-hidden">
        <div className={`h-full rounded-full bg-gradient-to-r ${barColor}`} style={{ width: `${Math.min(value, 100)}%` }} />
      </div>
      <span className="text-xs text-stone-400 font-mono">{value}%</span>
    </div>
  )
}

export default function CoursesAVenir() {
  const [courses, setCourses] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getCoursesAVenir(100).then(setCourses).catch(() => setCourses([])).finally(() => setLoading(false))
  }, [])

  if (loading) return (
    <div className="loading-pulse">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-racing-200 border-t-racing-600 rounded-full animate-spin" />
        <span className="text-sm text-stone-400">Chargement...</span>
      </div>
    </div>
  )

  const grouped = {}
  courses.forEach(c => { const key = new Date(c.date).toLocaleDateString('fr-FR'); if (!grouped[key]) grouped[key] = []; grouped[key].push(c) })

  return (
    <div className="space-y-8 animate-slide-up">
      <div className="flex items-center gap-3">
        <div className="w-10 h-10 rounded-xl bg-racing-50 flex items-center justify-center">
          <CalendarDays className="w-5 h-5 text-racing-600" />
        </div>
        <div>
          <h1 className="text-3xl font-display font-bold text-stone-900">Courses a venir</h1>
          <p className="text-stone-400 text-sm mt-0.5">Prochaines courses avec predictions</p>
        </div>
      </div>

      {courses.length === 0 ? (
        <div className="card p-8 text-center border-gold-200 bg-gold-50/50">
          <CalendarDays className="w-8 h-8 text-gold-500 mx-auto mb-3" />
          <p className="text-gold-700 font-medium">Aucune course a venir</p>
          <p className="text-stone-400 text-sm mt-1">Utilisez l'onglet Administration pour collecter les courses du jour.</p>
        </div>
      ) : (
        Object.entries(grouped).map(([dateStr, group]) => (
          <div key={dateStr}>
            <div className="flex items-center gap-3 mb-4">
              <div className="h-px flex-1 bg-stone-200" />
              <h2 className="text-sm font-semibold text-stone-500 uppercase tracking-wider">{dateStr}</h2>
              <div className="h-px flex-1 bg-stone-200" />
            </div>
            <div className="grid gap-3">
              {group.map(c => (
                <Link key={c.id} to={`/course/${c.id}`} className="card-hover p-5 flex items-center justify-between group">
                  <div className="flex items-center gap-5">
                    <div className="text-center min-w-[64px]">
                      <div className="text-xl font-mono font-bold text-racing-700">
                        {new Date(c.date).toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}
                      </div>
                      <div className="text-[10px] text-stone-400 font-mono uppercase tracking-wider mt-0.5">R{c.numero_reunion}C{c.numero_course}</div>
                    </div>
                    <div className="h-10 w-px bg-stone-200" />
                    <div>
                      <div className="flex items-center gap-2">
                        <MapPin className="w-3.5 h-3.5 text-stone-400" />
                        <span className="font-semibold text-stone-800 group-hover:text-racing-700 transition-colors">{c.hippodrome}</span>
                      </div>
                      <div className="flex items-center gap-3 mt-1 text-xs text-stone-400">
                        <span>{c.distance}m</span>
                        <span className="flex items-center gap-1"><Users className="w-3 h-3" /> {c.nombre_partants} partants</span>
                      </div>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    {c.favori_nom ? (
                      <>
                        <div className="text-right">
                          <div className="text-sm font-semibold text-racing-600">{c.favori_nom}</div>
                          <div className="text-[10px] text-stone-400 uppercase tracking-wider">Favori modele</div>
                        </div>
                        {c.favori_confiance != null && <ConfidenceBar value={c.favori_confiance} />}
                      </>
                    ) : <span className="text-xs text-stone-400 italic">Pas de prediction</span>}
                    <ChevronRight className="w-4 h-4 text-stone-300 group-hover:text-stone-500 transition-colors" />
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

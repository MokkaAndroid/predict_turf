import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getCourseDetail } from '../api'
import { ArrowLeft, Timer, Ruler, Trophy, Medal } from 'lucide-react'

function Badge({ type }) {
  const styles = {
    gagnant: 'bg-green-100 text-green-700 border-green-200',
    place: 'bg-gold-100 text-gold-700 border-gold-200',
    perdu: 'bg-stone-100 text-stone-500 border-stone-200',
    value: 'bg-purple-100 text-purple-600 border-purple-200',
  }
  const labels = { gagnant: 'Gagnant', place: 'Place', perdu: 'Perdu', value: 'Value Bet' }
  return <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-lg border ${styles[type]}`}>{labels[type]}</span>
}

function PodiumCard({ pred, rank }) {
  const border = { 1: 'border-gold-300 bg-gold-50/30', 2: 'border-stone-300 bg-stone-50/50', 3: 'border-gold-200 bg-gold-50/20' }
  const badge = { 1: 'bg-gradient-to-br from-gold-400 to-gold-500 text-white', 2: 'bg-gradient-to-br from-stone-300 to-stone-400 text-white', 3: 'bg-gradient-to-br from-gold-600 to-gold-700 text-white' }

  return (
    <div className={`card p-5 border-2 ${border[rank] || 'border-stone-200'}`}>
      <div className="flex justify-between items-start">
        <div>
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-display font-bold text-sm ${badge[rank] || 'bg-stone-200 text-stone-500'}`}>{rank}</div>
          <h3 className="font-display font-bold text-stone-800 text-lg mt-3">{pred.cheval_nom}</h3>
          <p className="text-xs text-stone-400 font-mono mt-0.5">N{pred.numero}</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-mono font-bold text-racing-600">{(pred.probabilite * 100).toFixed(0)}%</div>
          <div className="text-[10px] text-stone-400 uppercase tracking-wider">probabilite</div>
        </div>
      </div>
      <div className="mt-4 flex items-center gap-2">
        <div className="flex-1 h-1.5 bg-stone-200 rounded-full overflow-hidden">
          <div className="h-full bg-gradient-to-r from-racing-600 to-racing-500 rounded-full" style={{ width: `${Math.min(pred.score_confiance, 100)}%` }} />
        </div>
        <span className="text-xs text-stone-400 font-mono">{pred.score_confiance}%</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {pred.is_value_bet && <Badge type="value" />}
      </div>
      {pred.commentaire && <p className="text-xs text-stone-500 mt-3 italic bg-stone-50 rounded-lg px-3 py-2 border border-stone-100">{pred.commentaire}</p>}
    </div>
  )
}

export default function CourseDetail() {
  const { id } = useParams()
  const [course, setCourse] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => { getCourseDetail(id).then(setCourse).catch(() => setCourse(null)).finally(() => setLoading(false)) }, [id])

  if (loading) return (
    <div className="loading-pulse">
      <div className="flex flex-col items-center gap-3">
        <div className="w-8 h-8 border-2 border-racing-200 border-t-racing-600 rounded-full animate-spin" />
        <span className="text-sm text-stone-400">Chargement...</span>
      </div>
    </div>
  )

  if (!course) return (
    <div className="text-center py-20">
      <p className="text-red-500 font-medium">Course non trouvee</p>
      <Link to="/" className="text-sm text-racing-600 hover:text-racing-700 mt-2 inline-block">Retour au dashboard</Link>
    </div>
  )

  const isTerminee = course.statut === 'TERMINE'
  const predMap = {}
  course.predictions.forEach(p => { predMap[p.partant_id] = p })

  return (
    <div className="space-y-6 animate-slide-up">
      <Link to={isTerminee ? '/passees' : '/a-venir'} className="inline-flex items-center gap-2 text-sm text-stone-400 hover:text-racing-600 transition-colors">
        <ArrowLeft className="w-4 h-4" /> Retour
      </Link>

      <div className="card p-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-display font-bold text-stone-900">{course.hippodrome}</h1>
            <div className="flex items-center gap-4 mt-2 text-sm text-stone-400">
              <span className="font-mono">R{course.numero_reunion}C{course.numero_course}</span>
              <span className="flex items-center gap-1"><Timer className="w-3.5 h-3.5" /> {new Date(course.date).toLocaleString('fr-FR')}</span>
              <span className="flex items-center gap-1"><Ruler className="w-3.5 h-3.5" /> {course.distance}m</span>
              <span className="text-stone-600 font-medium">{course.discipline}</span>
            </div>
            {course.categorie && <p className="text-xs text-stone-400 mt-2">{course.categorie}</p>}
          </div>
          <span className={`px-3 py-1.5 rounded-xl text-xs font-bold uppercase tracking-wider ${isTerminee ? 'bg-stone-100 text-stone-500' : 'bg-green-100 text-green-700'}`}>
            {isTerminee ? 'Terminee' : 'A venir'}
          </span>
        </div>
        {course.dotation && (
          <div className="mt-4 pt-4 border-t border-stone-100">
            <span className="text-xs text-stone-400">Dotation: </span>
            <span className="text-sm font-mono font-semibold text-gold-600">{(course.dotation / 100).toLocaleString('fr-FR')} EUR</span>
          </div>
        )}
      </div>

      {course.predictions.length > 0 && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-lg bg-gold-100 flex items-center justify-center"><Trophy className="w-4 h-4 text-gold-600" /></div>
            <h2 className="font-display font-semibold text-stone-800 text-lg">Predictions du modele</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {course.predictions.slice(0, 3).map((pred, i) => <PodiumCard key={pred.partant_id} pred={pred} rank={i + 1} />)}
          </div>
        </div>
      )}

      <div className="card overflow-hidden">
        <div className="p-6 pb-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-stone-100 flex items-center justify-center"><Medal className="w-4 h-4 text-stone-500" /></div>
          <h2 className="font-display font-semibold text-stone-800 text-lg">Partants</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full table-premium">
            <thead>
              <tr>
                <th>N</th><th>Cheval</th><th>Jockey</th><th>Entraineur</th><th className="text-right">Poids</th><th className="text-right">Cote</th>
                {isTerminee && <><th className="text-center">Class.</th><th className="text-right">Rapp. Gagnant</th><th className="text-right">Rapp. Place</th></>}
                <th className="text-center">Prediction</th>
              </tr>
            </thead>
            <tbody>
              {course.partants.map(p => {
                const pred = predMap[p.id]
                const isGagnant = p.classement === 1
                const isPlace = p.classement && p.classement >= 1 && p.classement <= 3
                return (
                  <tr key={p.id} className={isGagnant ? '!bg-green-50' : isPlace ? '!bg-gold-50/50' : ''}>
                    <td className="font-mono font-bold text-stone-700">{p.numero}</td>
                    <td className="font-semibold text-stone-800">{p.cheval_nom}</td>
                    <td className="text-stone-500">{p.jockey_nom || <span className="text-stone-300">-</span>}</td>
                    <td className="text-stone-500">{p.entraineur_nom || <span className="text-stone-300">-</span>}</td>
                    <td className="text-right text-stone-400">{p.poids ? `${p.poids}kg` : '-'}</td>
                    <td className="text-right font-mono font-semibold text-stone-700">{p.cote_depart || p.cote_probable || '-'}</td>
                    {isTerminee && (
                      <>
                        <td className="text-center">
                          {p.classement ? <span className={`font-mono font-bold ${isGagnant ? 'text-gold-500' : isPlace ? 'text-gold-400' : 'text-stone-400'}`}>{p.classement}</span> : <span className="text-stone-300">-</span>}
                        </td>
                        <td className="text-right font-mono text-stone-600">{p.rapport_gagnant ? `${p.rapport_gagnant.toFixed(2)}` : '-'}</td>
                        <td className="text-right font-mono text-stone-600">{p.rapport_place ? `${p.rapport_place.toFixed(2)}` : '-'}</td>
                      </>
                    )}
                    <td className="text-center">
                      {pred ? (
                        <div className="flex items-center justify-center gap-2">
                          <span className="text-xs font-mono font-bold text-stone-700">#{pred.rang_predit}</span>
                          <span className="text-[10px] text-stone-400">({(pred.probabilite * 100).toFixed(0)}%)</span>
                          {pred.is_value_bet && <Badge type="value" />}
                        </div>
                      ) : <span className="text-stone-300">-</span>}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

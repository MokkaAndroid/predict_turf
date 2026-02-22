import { useState, useEffect } from 'react'
import { useParams, Link } from 'react-router-dom'
import { getCourseDetail } from '../api'
import { ArrowLeft, MapPin, Timer, Ruler, Trophy, Star, Zap, Medal } from 'lucide-react'

function Badge({ type }) {
  const styles = {
    gagnant: 'bg-racing-800/50 text-racing-300 border-racing-700/50',
    place: 'bg-gold-400/10 text-gold-400 border-gold-400/20',
    perdu: 'bg-dark-800 text-dark-400 border-dark-700',
    value: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  }
  const labels = { gagnant: 'Gagnant', place: 'Place', perdu: 'Perdu', value: 'Value Bet' }
  return (
    <span className={`text-[10px] font-bold uppercase tracking-wider px-2.5 py-1 rounded-lg border ${styles[type]}`}>
      {labels[type]}
    </span>
  )
}

function PodiumCard({ pred, rank }) {
  const rankStyles = {
    1: 'border-gold-400/40 bg-gradient-to-b from-gold-400/5 to-transparent',
    2: 'border-dark-400/30 bg-gradient-to-b from-dark-400/5 to-transparent',
    3: 'border-gold-700/30 bg-gradient-to-b from-gold-700/5 to-transparent',
  }
  const rankBg = {
    1: 'bg-gradient-to-br from-gold-400 to-gold-500 text-racing-950',
    2: 'bg-gradient-to-br from-dark-300 to-dark-400 text-dark-900',
    3: 'bg-gradient-to-br from-gold-700 to-gold-800 text-gold-200',
  }

  return (
    <div className={`card-glass p-5 border-2 ${rankStyles[rank] || 'border-dark-700/50'}`}>
      <div className="flex justify-between items-start">
        <div>
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center font-display font-bold text-sm ${rankBg[rank] || 'bg-dark-800 text-dark-400'}`}>
            {rank}
          </div>
          <h3 className="font-display font-bold text-white text-lg mt-3">{pred.cheval_nom}</h3>
          <p className="text-xs text-dark-400 font-mono mt-0.5">N{pred.numero}</p>
        </div>
        <div className="text-right">
          <div className="text-2xl font-mono font-bold text-racing-400">
            {(pred.probabilite * 100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-dark-500 uppercase tracking-wider">probabilite</div>
        </div>
      </div>
      <div className="mt-4 flex items-center gap-2">
        <div className="flex-1 h-1.5 bg-dark-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-racing-500 to-racing-400 rounded-full"
            style={{ width: `${Math.min(pred.score_confiance, 100)}%` }}
          />
        </div>
        <span className="text-xs text-dark-400 font-mono">{pred.score_confiance}%</span>
      </div>
      <div className="mt-3 flex flex-wrap gap-2">
        {pred.is_value_bet && <Badge type="value" />}
      </div>
      {pred.commentaire && (
        <p className="text-xs text-dark-400 mt-3 italic bg-dark-800/30 rounded-lg px-3 py-2 border border-dark-700/30">
          {pred.commentaire}
        </p>
      )}
    </div>
  )
}

export default function CourseDetail() {
  const { id } = useParams()
  const [course, setCourse] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    getCourseDetail(id)
      .then(setCourse)
      .catch(() => setCourse(null))
      .finally(() => setLoading(false))
  }, [id])

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

  if (!course) {
    return (
      <div className="text-center py-20">
        <p className="text-red-400 font-medium">Course non trouvee</p>
        <Link to="/" className="text-sm text-gold-400 hover:text-gold-300 mt-2 inline-block">Retour au dashboard</Link>
      </div>
    )
  }

  const isTerminee = course.statut === 'TERMINE'
  const predMap = {}
  course.predictions.forEach(p => { predMap[p.partant_id] = p })

  return (
    <div className="space-y-6 animate-slide-up">
      {/* Back link */}
      <Link
        to={isTerminee ? '/passees' : '/a-venir'}
        className="inline-flex items-center gap-2 text-sm text-dark-400 hover:text-gold-400 transition-colors"
      >
        <ArrowLeft className="w-4 h-4" />
        Retour
      </Link>

      {/* Course header */}
      <div className="card-glass p-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-display font-bold text-white">{course.hippodrome}</h1>
            <div className="flex items-center gap-4 mt-2 text-sm text-dark-400">
              <span className="font-mono">R{course.numero_reunion}C{course.numero_course}</span>
              <span className="flex items-center gap-1">
                <Timer className="w-3.5 h-3.5" />
                {new Date(course.date).toLocaleString('fr-FR')}
              </span>
              <span className="flex items-center gap-1">
                <Ruler className="w-3.5 h-3.5" />
                {course.distance}m
              </span>
              <span className="text-dark-300 font-medium">{course.discipline}</span>
            </div>
            {course.categorie && (
              <p className="text-xs text-dark-500 mt-2">{course.categorie}</p>
            )}
          </div>
          <span className={`px-3 py-1.5 rounded-xl text-xs font-bold uppercase tracking-wider ${
            isTerminee
              ? 'bg-dark-800 text-dark-400 border border-dark-700'
              : 'bg-racing-800/50 text-racing-300 border border-racing-700/50'
          }`}>
            {isTerminee ? 'Terminee' : 'A venir'}
          </span>
        </div>

        {course.dotation && (
          <div className="mt-4 pt-4 border-t border-dark-800">
            <span className="text-xs text-dark-500">Dotation: </span>
            <span className="text-sm font-mono font-semibold text-gold-400">
              {(course.dotation / 100).toLocaleString('fr-FR')} EUR
            </span>
          </div>
        )}
      </div>

      {/* Predictions Podium */}
      {course.predictions.length > 0 && (
        <div>
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-lg bg-gold-400/10 flex items-center justify-center">
              <Trophy className="w-4 h-4 text-gold-400" />
            </div>
            <h2 className="font-display font-semibold text-white text-lg">Predictions du modele</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-4">
            {course.predictions.slice(0, 3).map((pred, i) => (
              <PodiumCard key={pred.partant_id} pred={pred} rank={i + 1} />
            ))}
          </div>
        </div>
      )}

      {/* Participants table */}
      <div className="card-glass overflow-hidden">
        <div className="p-6 pb-3 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-dark-800 flex items-center justify-center">
            <Medal className="w-4 h-4 text-dark-300" />
          </div>
          <h2 className="font-display font-semibold text-white text-lg">Partants</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full table-premium">
            <thead>
              <tr>
                <th>N</th>
                <th>Cheval</th>
                <th>Jockey</th>
                <th>Entraineur</th>
                <th className="text-right">Poids</th>
                <th className="text-right">Cote</th>
                {isTerminee && (
                  <>
                    <th className="text-center">Class.</th>
                    <th className="text-right">Rapp. Gagnant</th>
                    <th className="text-right">Rapp. Place</th>
                  </>
                )}
                <th className="text-center">Prediction</th>
              </tr>
            </thead>
            <tbody>
              {course.partants.map(p => {
                const pred = predMap[p.id]
                const isGagnant = p.classement === 1
                const isPlace = p.classement && p.classement >= 1 && p.classement <= 3
                return (
                  <tr
                    key={p.id}
                    className={
                      isGagnant ? '!bg-racing-900/20' : isPlace ? '!bg-gold-400/5' : ''
                    }
                  >
                    <td className="font-mono font-bold text-dark-200">{p.numero}</td>
                    <td className="font-semibold text-white">{p.cheval_nom}</td>
                    <td className="text-dark-300">{p.jockey_nom || <span className="text-dark-600">-</span>}</td>
                    <td className="text-dark-300">{p.entraineur_nom || <span className="text-dark-600">-</span>}</td>
                    <td className="text-right text-dark-400">{p.poids ? `${p.poids}kg` : '-'}</td>
                    <td className="text-right font-mono font-semibold text-dark-200">{p.cote_depart || p.cote_probable || '-'}</td>
                    {isTerminee && (
                      <>
                        <td className="text-center">
                          {p.classement ? (
                            <span className={`font-mono font-bold ${
                              isGagnant ? 'text-gold-400' : isPlace ? 'text-gold-500/60' : 'text-dark-500'
                            }`}>
                              {p.classement}
                            </span>
                          ) : <span className="text-dark-600">-</span>}
                        </td>
                        <td className="text-right font-mono text-dark-300">
                          {p.rapport_gagnant ? `${p.rapport_gagnant.toFixed(2)}` : '-'}
                        </td>
                        <td className="text-right font-mono text-dark-300">
                          {p.rapport_place ? `${p.rapport_place.toFixed(2)}` : '-'}
                        </td>
                      </>
                    )}
                    <td className="text-center">
                      {pred ? (
                        <div className="flex items-center justify-center gap-2">
                          <span className="text-xs font-mono font-bold text-dark-200">#{pred.rang_predit}</span>
                          <span className="text-[10px] text-dark-500">({(pred.probabilite * 100).toFixed(0)}%)</span>
                          {pred.is_value_bet && <Badge type="value" />}
                        </div>
                      ) : <span className="text-dark-600">-</span>}
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

const BASE = '/api'

async function fetchJSON(url, options = {}) {
  const res = await fetch(`${BASE}${url}`, options)
  if (!res.ok) throw new Error(`HTTP ${res.status}`)
  return res.json()
}

export async function getCoursesPassees(limit = 50, offset = 0, hippodrome = '') {
  const params = new URLSearchParams({ limit, offset })
  if (hippodrome) params.set('hippodrome', hippodrome)
  return fetchJSON(`/courses/passees?${params}`)
}

export async function getCoursesAVenir(limit = 50) {
  return fetchJSON(`/courses/a-venir?limit=${limit}`)
}

export async function getCourseDetail(id) {
  return fetchJSON(`/course/${id}`)
}

export async function getBacktestingStats(mise = 1) {
  return fetchJSON(`/backtesting/stats?mise=${mise}`)
}

export async function getPrevisionsJour() {
  return fetchJSON('/previsions/jour')
}

export async function getHealth() {
  return fetchJSON('/health')
}

export async function postCollect(start, end) {
  return fetchJSON(`/collect?start=${start}&end=${end}`, { method: 'POST' })
}

export async function postCollectToday() {
  return fetchJSON('/collect/today', { method: 'POST' })
}

export async function postTrain() {
  // Lance l'entraînement en arrière-plan
  const start = await fetchJSON('/train', { method: 'POST' })
  if (start.status === 'already_running') {
    // Si déjà en cours, on attend quand même le résultat
  } else if (start.status !== 'started') {
    return start
  }

  // Poll le statut toutes les 2 secondes jusqu'à la fin
  while (true) {
    await new Promise(r => setTimeout(r, 2000))
    const status = await fetchJSON('/train/status')
    if (status.state === 'done') {
      return status.result
    }
    // state === 'running' → on continue de poller
  }
}

export async function postPredict() {
  const start = await fetchJSON('/predict', { method: 'POST' })
  if (start.status !== 'started' && start.status !== 'already_running') {
    return start
  }

  while (true) {
    await new Promise(r => setTimeout(r, 2000))
    const status = await fetchJSON('/predict/status')
    if (status.state === 'done') {
      return status.result
    }
  }
}

export async function postBacktest() {
  return fetchJSON('/backtest', { method: 'POST' })
}

export async function postReset() {
  return fetchJSON('/reset', { method: 'POST' })
}

export async function getBilanVeille() {
  return fetchJSON('/bilan-veille')
}

export async function getConfianceStats() {
  return fetchJSON('/confiance/stats')
}

export async function postDailyUpdate() {
  return fetchJSON('/daily-update', { method: 'POST' })
}

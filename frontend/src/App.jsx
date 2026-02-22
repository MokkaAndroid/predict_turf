import { Routes, Route, NavLink } from 'react-router-dom'
import { LayoutDashboard, Target, Clock, CalendarDays, Settings, Trophy } from 'lucide-react'
import Dashboard from './pages/Dashboard'
import CoursesPassees from './pages/CoursesPassees'
import CoursesAVenir from './pages/CoursesAVenir'
import PrevisionsJour from './pages/PrevisionsJour'
import CourseDetail from './pages/CourseDetail'
import Admin from './pages/Admin'

const navItems = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/jour', label: 'Previsions du jour', icon: Target },
  { to: '/passees', label: 'Historique', icon: Clock },
  { to: '/a-venir', label: 'Courses a venir', icon: CalendarDays },
  { to: '/admin', label: 'Administration', icon: Settings },
]

export default function App() {
  return (
    <div className="min-h-screen bg-stone-50 text-stone-900">
      {/* Navigation */}
      <nav className="bg-racing-800 sticky top-0 z-50 shadow-lg">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center h-16">
            {/* Logo */}
            <NavLink to="/" className="flex items-center gap-3 mr-10 group">
              <div className="w-9 h-9 rounded-xl bg-white/10 flex items-center justify-center group-hover:bg-white/20 transition-colors">
                <Trophy className="w-5 h-5 text-gold-400" />
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-display font-bold text-white tracking-tight leading-tight">
                  Hippique
                </span>
                <span className="text-[10px] font-medium text-gold-400/80 uppercase tracking-widest leading-tight">
                  Previsions IA
                </span>
              </div>
            </NavLink>

            {/* Nav Links */}
            <div className="flex items-center gap-1">
              {navItems.map(({ to, label, icon: Icon }) => (
                <NavLink
                  key={to}
                  to={to}
                  end={to === '/'}
                  className={({ isActive }) =>
                    `flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium transition-all duration-200 ${
                      isActive
                        ? 'bg-white/15 text-gold-300'
                        : 'text-white/70 hover:text-white hover:bg-white/10'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden lg:inline">{label}</span>
                </NavLink>
              ))}
            </div>

            {/* Right side */}
            <div className="ml-auto flex items-center gap-3">
              <div className="hidden md:flex items-center gap-2 text-xs text-white/60">
                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                <span>Modele actif</span>
              </div>
            </div>
          </div>
        </div>
        <div className="h-px bg-gradient-to-r from-transparent via-gold-400/40 to-transparent" />
      </nav>

      {/* Main content */}
      <main className="max-w-7xl mx-auto px-6 py-8 animate-fade-in">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/jour" element={<PrevisionsJour />} />
          <Route path="/passees" element={<CoursesPassees />} />
          <Route path="/a-venir" element={<CoursesAVenir />} />
          <Route path="/course/:id" element={<CourseDetail />} />
          <Route path="/admin" element={<Admin />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="border-t border-stone-200 mt-12 bg-white">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-stone-400">
            <Trophy className="w-3.5 h-3.5 text-gold-500" />
            <span className="font-display font-semibold">Hippique</span>
            <span>&middot;</span>
            <span>Previsions intelligentes par IA</span>
          </div>
          <div className="text-xs text-stone-400">
            Donnees PMU &middot; Modele LightGBM
          </div>
        </div>
      </footer>
    </div>
  )
}

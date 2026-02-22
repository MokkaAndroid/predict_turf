import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
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
  const location = useLocation()

  return (
    <div className="min-h-screen bg-dark-950 text-dark-100">
      {/* Navigation */}
      <nav className="bg-dark-900 border-b border-dark-800/80 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex items-center h-16">
            {/* Logo */}
            <NavLink to="/" className="flex items-center gap-3 mr-10 group">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-racing-600 to-racing-700 flex items-center justify-center shadow-md group-hover:shadow-lg transition-shadow">
                <Trophy className="w-5 h-5 text-gold-400" />
              </div>
              <div className="flex flex-col">
                <span className="text-lg font-display font-bold text-white tracking-tight leading-tight">
                  Hippique
                </span>
                <span className="text-[10px] font-medium text-gold-400/80 uppercase tracking-widest leading-tight">
                  Previsions
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
                        ? 'bg-racing-700/60 text-gold-300 shadow-inner-glow'
                        : 'text-dark-300 hover:text-white hover:bg-dark-800/60'
                    }`
                  }
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden lg:inline">{label}</span>
                </NavLink>
              ))}
            </div>

            {/* Right side accent */}
            <div className="ml-auto flex items-center gap-3">
              <div className="hidden md:flex items-center gap-2 text-xs text-dark-400">
                <div className="w-2 h-2 rounded-full bg-racing-500 animate-pulse" />
                <span>Modele actif</span>
              </div>
            </div>
          </div>
        </div>
        {/* Gold accent line */}
        <div className="h-px bg-gradient-to-r from-transparent via-gold-400/30 to-transparent" />
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
      <footer className="border-t border-dark-800/50 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-2 text-xs text-dark-500">
            <Trophy className="w-3.5 h-3.5 text-gold-400/50" />
            <span className="font-display">Hippique</span>
            <span>&middot;</span>
            <span>Previsions intelligentes par IA</span>
          </div>
          <div className="text-xs text-dark-600">
            Donnees PMU &middot; Modele LightGBM
          </div>
        </div>
      </footer>
    </div>
  )
}

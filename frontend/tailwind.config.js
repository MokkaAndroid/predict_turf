/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        racing: {
          50: '#f0f7f2',
          100: '#d4e8d9',
          200: '#a8d1b3',
          300: '#6db380',
          400: '#3d8a55',
          500: '#1e6b3a',
          600: '#165a30',
          700: '#0f4824',
          800: '#0a3219',
          900: '#072410',
          950: '#041a0b',
        },
        gold: {
          50: '#fdf9ef',
          100: '#f9f0d4',
          200: '#f2dfa6',
          300: '#e8c96e',
          400: '#d4af37',
          500: '#c9a027',
          600: '#b08a1e',
          700: '#8f6d18',
          800: '#755713',
          900: '#5c4410',
        },
        dark: {
          50: '#f5f5f6',
          100: '#e5e5e8',
          200: '#cdcdd3',
          300: '#a9a9b3',
          400: '#7e7e8c',
          500: '#636371',
          600: '#4d4d58',
          700: '#3d3d46',
          800: '#2a2a31',
          900: '#1a1a1f',
          950: '#111114',
        },
      },
      fontFamily: {
        display: ['Playfair Display', 'Georgia', 'serif'],
        body: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'gradient-racing': 'linear-gradient(135deg, #072410 0%, #0a3219 50%, #0f4824 100%)',
        'gradient-gold': 'linear-gradient(135deg, #d4af37 0%, #e8c96e 50%, #d4af37 100%)',
        'gradient-card': 'linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0) 100%)',
      },
      boxShadow: {
        'premium': '0 4px 24px -4px rgba(0, 0, 0, 0.12), 0 0 0 1px rgba(0, 0, 0, 0.04)',
        'premium-lg': '0 8px 40px -8px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(0, 0, 0, 0.06)',
        'gold': '0 4px 24px -4px rgba(212, 175, 55, 0.25)',
        'inner-glow': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.05)',
      },
      borderRadius: {
        '2xl': '1rem',
        '3xl': '1.5rem',
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.4s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
    },
  },
  plugins: [],
}

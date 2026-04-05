/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          DEFAULT: '#1a1510',
          soft: '#2d2520',
          muted: '#4a3f35',
          dim: '#7a6b5e',
          faint: '#a89585',
          ghost: '#c4b4a6',
        },
        parchment: {
          DEFAULT: '#f5efe6',
          warm: '#ede4d8',
          deep: '#e0d4c4',
          darker: '#cfc0ae',
        },
        terra: {
          DEFAULT: '#c4714a',
          light: '#d4896a',
          dim: '#c4714a18',
          glow: '#c4714a30',
        },
        sage: {
          DEFAULT: '#7a9e8a',
          light: '#93b5a0',
          dim: '#7a9e8a15',
        },
        clay: {
          DEFAULT: '#b8956a',
          light: '#ccaa82',
          dim: '#b8956a15',
        },
        rose: {
          DEFAULT: '#c47878',
          dim: '#c4787815',
        },
      },
      fontFamily: {
        display: ['Lora', 'Georgia', 'serif'],
        body: ['DM Sans', 'system-ui', 'sans-serif'],
      },
      fontSize: {
        '2xs': ['0.65rem', { lineHeight: '1rem' }],
      },
      boxShadow: {
        'soft': '0 2px 20px rgba(26,21,16,0.08)',
        'card': '0 4px 32px rgba(26,21,16,0.12)',
        'warm': '0 8px 40px rgba(196,113,74,0.12)',
        'inner-warm': 'inset 0 1px 0 rgba(245,239,230,0.6)',
      },
      borderRadius: {
        'xl': '12px',
        '2xl': '18px',
        '3xl': '26px',
        '4xl': '36px',
      },
      backgroundImage: {
        'grain': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.04'/%3E%3C/svg%3E\")",
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'fade-up': 'fadeUp 0.5s ease-out',
        'slide-up': 'slideUp 0.35s ease-out',
        'breathe': 'breathe 4s ease-in-out infinite',
        'shimmer': 'shimmer 1.8s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(16px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        breathe: {
          '0%, 100%': { opacity: '0.4', transform: 'scale(1)' },
          '50%': { opacity: '0.7', transform: 'scale(1.02)' },
        },
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
      },
    },
  },
  plugins: [],
}

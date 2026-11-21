/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        studio: {
          bg: '#E5E8CE',
          bgSecondary: '#EEF0DC',
          card: '#F2F3E2',
          cardElevated: '#FAFBF4',
          accent: '#789575',
          accentHover: '#668263',
          accentSecondary: '#A7B99A',
          text: '#263629',
          textMuted: '#647B62',
          border: '#CCD3B5',
          borderLight: '#DCE2C8',
          plotBg: '#F5F5E9',
          warning: '#A67824',
          warningBg: '#F7EED9',
          error: '#9C4141',
          errorBg: '#F9E6E6',
          success: '#36663E',
          successBg: '#E4F0E6',
        },
      },
      fontFamily: {
        sans: ['-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'Oxygen', 'Ubuntu', 'sans-serif'],
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
      },
      boxShadow: {
        'subtle': '0 1px 3px 0 rgba(38, 54, 41, 0.05), 0 1px 2px -1px rgba(38, 54, 41, 0.03)',
        'card': '0 2px 6px -1px rgba(38, 54, 41, 0.07), 0 1px 4px -2px rgba(38, 54, 41, 0.04)',
        'dropdown': '0 4px 12px -2px rgba(38, 54, 41, 0.12), 0 2px 6px -2px rgba(38, 54, 41, 0.08)',
      },
      borderRadius: {
        'xs': '4px',
        'sm': '6px',
        'md': '8px',
        'lg': '10px',
        'xl': '12px',
        '2xl': '16px',
      }
    },
  },
  plugins: [],
}

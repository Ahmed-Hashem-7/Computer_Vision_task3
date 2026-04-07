import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Django serves built assets under STATIC_URL (static/)
  base: '/static/',
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Vite config with a dev proxy so calls to /api are forwarded to the backend
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})

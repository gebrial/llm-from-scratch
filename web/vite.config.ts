import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // The API's CORS allow-list names this exact origin, port included, so
  // fail rather than quietly move to another port when 5173 is taken.
  server: { port: 5173, strictPort: true },
})

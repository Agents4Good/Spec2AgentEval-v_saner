import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ command }) => ({
  plugins: [react()],
  // GitHub Pages serves the site under /<repo>/
  base: command === 'build' ? '/Spec2AgentEval-v_saner/' : '/',
}))

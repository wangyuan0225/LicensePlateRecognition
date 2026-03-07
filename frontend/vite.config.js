import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    },
  },
  server: {
    allowedHosts: ['elect-widescreen-atomic-guys.trycloudflare.com', '1d316bcb.r15.vip.cpolar.cn', 'app.wangyuan0225.org', 'www.wangyuan0225.org'],
    proxy: {
      '/api': {
        target: 'http://localhost:8088',
        changeOrigin: true,
      },
      '/static': {
        target: 'http://localhost:8088',
        changeOrigin: true,
      },
    },
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        host: true, // Listen on all addresses, including LAN and public
        port: 3000,
        allowedHosts: 'all',
        proxy: {
            '/api': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            },
            '/output': {
                target: 'http://localhost:8000',
                changeOrigin: true,
            }
        }
    }
})

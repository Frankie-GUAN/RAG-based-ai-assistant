import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './styles/main.css'
import { initGSAP } from './composables/useGSAP'

const app = createApp(App)
app.use(createPinia())
app.use(router)

initGSAP()  // register ScrollTrigger + matchMedia

app.mount('#app')

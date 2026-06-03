import { gsap } from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

let initialized = false

export function initGSAP() {
  if (initialized) return
  gsap.registerPlugin(ScrollTrigger)

  // Global reduced-motion support
  const mm = gsap.matchMedia()
  mm.add('(prefers-reduced-motion: reduce)', () => {
    gsap.defaults({ duration: 0, overwrite: true })
  })

  initialized = true
}

export { gsap, ScrollTrigger }

// Types the VITE_ variables from web/.env.*, so import.meta.env.VITE_API_URL
// is a string rather than `any`.
interface ImportMetaEnv {
  readonly VITE_API_URL: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

// The API's base URL, from VITE_API_URL in web/.env.development or
// web/.env.production depending on the mode Vite runs in.

const apiUrl = import.meta.env.VITE_API_URL

// A missing env file would otherwise send every request to "undefined/...",
// which fails far from the cause. Better to fail on load, naming the variable.
if (!apiUrl) {
  throw new Error("VITE_API_URL is not set; see web/.env.development")
}

// Trimmed so "https://host/" and "https://host" both join as `${API_URL}/path`.
export const API_URL: string = apiUrl.replace(/\/+$/, "")

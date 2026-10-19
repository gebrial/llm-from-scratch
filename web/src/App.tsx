import { useEffect, useState } from 'react'
import StoriesFeed from './StoriesFeed'
import { API_URL } from './api'
import GenerateForm from './GenerateForm'

export type ServerStatus = "checking" | "ready" | "asleep"
type AbortReason = "timeout" | "cleanup"

function App() {
  const [serverStatus, setServerStatus] = useState<ServerStatus>("checking")
  const [feedRefreshKey, setFeedRefreshKey] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    fetch(`${API_URL}/ping`, {signal: controller.signal})
      .then(res => res.json())
      .then(status => {
        if (status.status === "ok") setServerStatus("ready")
        else setServerStatus("asleep")
      })
      .catch(() => {
        if (controller.signal.reason !== ("cleanup" satisfies AbortReason)) setServerStatus("asleep")
      })
      .finally(() => {
        clearTimeout(timeoutId)
      })
    
    const timeoutId = setTimeout(() => {
      controller.abort("timeout" satisfies AbortReason);
    }, 3000)

    return () => {
      clearTimeout(timeoutId)
      controller.abort("cleanup" satisfies AbortReason)
    }
  }, [])

  return (
    <main className="app">
    <header className="app-header">
      <h1>TinyStories</h1>
      <p>A small GPT, trained from scratch, that tells short stories.</p>
    </header>
    {serverStatus === "ready" && <GenerateForm onGenerateStart={() => setFeedRefreshKey(k => k + 1)}/> }
    { 
      serverStatus === "checking" ? <p className="status">Checking</p> : 
      (serverStatus === "ready" ? <StoriesFeed refreshKey={feedRefreshKey} /> : <p className="status">Server Asleep</p>)
    }
    </main>
  )
}

export default App

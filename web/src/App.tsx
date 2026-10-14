import { useEffect, useState } from 'react'
import StoriesFeed from './StoriesFeed'
import { API_URL } from './api'
import GenerateForm from './GenerateForm'

type ServerStatus = "checking" | "ready" | "asleep"
type AbortReason = "timeout" | "cleanup"

function App() {
  const [serverStatus, setServerStatus] = useState<ServerStatus>("checking")

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
    <>
    <GenerateForm />
    { 
      serverStatus === "checking" ? <p>Checking</p> : 
      (serverStatus === "ready" ? <StoriesFeed /> : <p>Asleep</p>)
    }
    </>
  )
}

export default App

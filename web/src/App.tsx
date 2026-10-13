import { useEffect, useState } from 'react'

type ServerStatus = "checking" | "ready" | "asleep"
type AbortReason = "timeout" | "cleanup"

function App() {
  const [serverStatus, setServerStatus] = useState<ServerStatus>("checking")

  useEffect(() => {
    const controller = new AbortController()
    fetch("https://stories-api.gebrial.ca/ping", {signal: controller.signal})
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
    { 
      serverStatus === "checking" ? <p>Checking</p> : 
      (serverStatus === "ready" ? <p>Ready</p> : <p>Asleep</p>)
    }
    </>
  )
}

export default App

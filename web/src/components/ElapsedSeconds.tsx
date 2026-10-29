import { useEffect, useState } from "react"

// Seconds, to a tenth, since this component mounted. Mount it only while something is
// in progress, so the interval exists only then.
export default function ElapsedSeconds() {
  const [startedAt] = useState(() => Date.now())
  const [now, setNow] = useState(startedAt)

  useEffect(() => {
    const intervalId = setInterval(() => setNow(Date.now()), 100)
    return () => clearInterval(intervalId)
  }, [])

  return <span className="elapsed">{((now - startedAt) / 1000).toFixed(1)}s</span>
}

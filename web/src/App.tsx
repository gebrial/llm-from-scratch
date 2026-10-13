import { useEffect } from 'react'

function App() {
  useEffect(() => {
    fetch("https://stories-api.gebrial.ca/ping")
  }, [])

  return (
    <>

    </>
  )
}

export default App

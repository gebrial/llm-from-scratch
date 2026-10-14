import { useEffect, useState } from "react"

type Story = {
    prompt: string,
    story: string,
    created_at: string
}

type StoriesResponse = {
    stories: Story[]
}

type StoriesStatus = 
  | { status: "loading" }
  | { status: "ready", stories: Story[] }
  | { status: "failed" }

export default function StoriesFeed() {
  const [storiesStatus, setStoriesStatus] = useState<StoriesStatus>({status: "loading"})

  useEffect(() => {
    const controller = new AbortController()
    fetch("https://stories-api.gebrial.ca/stories", {signal: controller.signal})
      .then(res => {
        if (!res.ok) {
          throw(new Error(`get stories failed: ${res.status}`))
        }
        return res.json()
      })
      .then((res: StoriesResponse) => {
        setStoriesStatus({status: "ready", stories: res.stories})
      })
      .catch(err => {
        if (controller.signal.reason === "cleanup") return
        
        setStoriesStatus({status: "failed"})
        console.error(err)
      })
    
    return () => {
        controller.abort("cleanup")
    }
  }, [])

  if (storiesStatus.status === "loading") return <p>Stories loading...</p>
  if (storiesStatus.status === "failed") return <p>Couldn't load stories</p>
  if (storiesStatus.stories.length === 0) return <p>No Stories yet...</p>
  return (
    <>
      <p>Stories (bolded text is the prompt):</p>
      <ul>
        {storiesStatus.stories.map(story => {
          return (<li key={story.created_at}><b>{story.prompt}</b>{story.story.slice(story.prompt.length + 1)}</li>)
        })}
      </ul>
    </>
  )
}
import { useEffect, useState } from "react"
import { API_URL } from "./api"

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
    fetch(`${API_URL}/stories`, {signal: controller.signal})
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

  if (storiesStatus.status === "loading") return <p className="status">Stories loading...</p>
  if (storiesStatus.status === "failed") return <p className="status">Couldn't load stories</p>
  if (storiesStatus.stories.length === 0) return <p className="status">No Stories yet...</p>
  return (
    <>
      <h2 className="feed-heading">Recent stories</h2>
      <ul className="feed">
        {storiesStatus.stories.map(story => {
          return (<li className="story" key={story.created_at}><b className="story-prompt">{story.prompt}</b>{story.story.slice(story.prompt.length + 1)}</li>)
        })}
      </ul>
    </>
  )
}
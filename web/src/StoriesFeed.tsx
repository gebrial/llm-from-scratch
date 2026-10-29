import { useEffect, useState } from "react"
import { API_URL } from "./api"
import Story from "./Story"

type StoryRow = {
    prompt: string,
    story: string,
    created_at: string
}

type StoriesResponse = {
    stories: StoryRow[]
}

type StoriesStatus = 
  | { status: "loading" }
  | { status: "ready", stories: StoryRow[] }
  | { status: "failed" }

type StoriesFeedProps = {
  refreshKey: number,
}

export default function StoriesFeed({ refreshKey }: StoriesFeedProps) {
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
        
        // don't overwrite existing stories in the feed if the next fetch fails
        setStoriesStatus(prev => prev.status === "ready" ? prev : {status: "failed"})

        console.error(err)
      })
    
    return () => {
        controller.abort("cleanup")
    }
  // refreshKey: rerun when a new story is submitted
  }, [refreshKey])

  if (storiesStatus.status === "loading") return <p className="status">Stories loading...</p>
  if (storiesStatus.status === "failed") return <p className="status">Couldn't load stories</p>
  if (storiesStatus.stories.length === 0) return <p className="status">No Stories yet...</p>
  return (
    <>
      <h2 className="feed-heading">Recent stories</h2>
      <ul className="feed">
        {storiesStatus.stories.map(story => {
          // TODO use id for key instead of created_at datetime
          // TODO: instead of assuming where to slice the story, compare it with the prompt to compute the continuation
          const continuation = story.story.slice(story.prompt.length + 1)
          return (<li className="story" key={story.created_at}><Story prompt={story.prompt} continuation={continuation}/></li>)
        })}
      </ul>
    </>
  )
}
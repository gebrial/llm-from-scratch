import { useState } from "react"
import { API_URL } from "./api"

type GenerateStatus = 
  | { status: "none" }
  | { status: "generating", prompt: string }
  | { status: "finished", prompt: string, story: string}
  | { status: "error", error: string }

export default function GenerateForm() {
  // TODO: show generating state, and disable form while generating
  // TODO: differentiate error state (rate limited, timout, unknown/other)
  // TODO: abort generation after 30s
  // TODO: refresh StoriesFeed after new story returns
  const [promptInput, setPromptInput] = useState("")
  const [generateStatus, setGenerateStatus] = useState<GenerateStatus>({status: "none"})

  function generateStory(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!promptInput.trim()) return

    setGenerateStatus({status: "generating", prompt: promptInput})
    fetch(`${API_URL}/generate`, {
      method: "POST",
      body: JSON.stringify({ prompt: promptInput }),
      headers: { "Content-Type": "application/json" }
    })
    .then(res => {
      if (!res.ok) throw new Error(`generate post failed: ${res.status}`)
      return res.json()
    })
    .then((newStory: string) => {
      setGenerateStatus({status: "finished", prompt: promptInput, story: newStory})
    })
    .catch((err: unknown) => {
      console.error(err)
      setGenerateStatus({status: "error", error: String(err)})
    })
  }

  return (
    <>
      <form className="generate-form" onSubmit={generateStory}>
        <input 
          onChange={e => setPromptInput(e.target.value)}
          value={promptInput}
          placeholder="Story Prompt"
        />
        <button>Submit</button>
      </form>
      {
        generateStatus.status === "finished" &&
        <p className="story story--new">
          <b className="story-prompt">{generateStatus.prompt}</b>
          {generateStatus.story.slice(generateStatus.prompt.length + 1)}
        </p>
      }
    </>
  )
}
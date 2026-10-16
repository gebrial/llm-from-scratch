import { useState } from "react"
import { API_URL } from "./api"

type GenerateStatus = 
  | { status: "none" }
  | { status: "generating", prompt: string }
  | { status: "finished", prompt: string, story: string}
  | { status: "error", error: string }

export default function GenerateForm() {
  // TODO: differentiate error state (rate limited, timout, unknown/other)
  // TODO: abort generation after 30s
  // TODO: refresh StoriesFeed after new story returns
  // TODO: make StoryText component that renders a story with the prompt in different formatting
  // TODO: instead of assuming where to slice the story, compare it with the prompt to compute it
  const [promptInput, setPromptInput] = useState("")
  const [generateStatus, setGenerateStatus] = useState<GenerateStatus>({status: "none"})

  function generateStory(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!promptInput.trim()) return

    const promptInputTrimmed = promptInput.trim();
    setGenerateStatus({status: "generating", prompt: promptInputTrimmed})
    fetch(`${API_URL}/generate`, {
      method: "POST",
      body: JSON.stringify({ prompt: promptInputTrimmed }),
      headers: { "Content-Type": "application/json" }
    })
    .then(res => {
      if (!res.ok) throw new Error(`generate post failed: ${res.status}`)
      return res.json()
    })
    .then((newStory: string) => {
      setGenerateStatus({status: "finished", prompt: promptInputTrimmed, story: newStory})
      setPromptInput("")
    })
    .catch((err: unknown) => {
      console.error(err)
      setGenerateStatus({status: "error", error: String(err)})
    })
  }

  const isStoryGenerating = generateStatus.status === "generating"
  return (
    <>
      <form className="generate-form" onSubmit={generateStory}>
        <input 
          onChange={e => setPromptInput(e.target.value)}
          value={promptInput}
          placeholder="Story Prompt"
          disabled={isStoryGenerating}
        />
        <button disabled={isStoryGenerating}>{isStoryGenerating ? "Generating..." : "Submit"}</button>
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
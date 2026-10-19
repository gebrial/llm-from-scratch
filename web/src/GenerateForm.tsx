import { useState } from "react"
import { API_URL } from "./api"
import Story from "./Story"

type GenerateStatus = 
  | { status: "none" }
  | { status: "generating", prompt: string }
  | { status: "finished", prompt: string, story: string}
  | { status: "error", error: string }

type GenerateFormProps = {
  onGenerateStart: () => void,
}

export default function GenerateForm({ onGenerateStart }: GenerateFormProps) {
  // TODO: differentiate error state (rate limited, timout, unknown/other)
  // TODO: abort generation after 30s
  const [promptInput, setPromptInput] = useState("")
  const [generateStatus, setGenerateStatus] = useState<GenerateStatus>({status: "none"})

  function generateStory(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!promptInput.trim()) return

    const promptInputTrimmed = promptInput.trim();
    setGenerateStatus({status: "generating", prompt: promptInputTrimmed})
    onGenerateStart()
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
          <Story story={generateStatus.story} prompt={generateStatus.prompt} />
        </p>
      }
    </>
  )
}
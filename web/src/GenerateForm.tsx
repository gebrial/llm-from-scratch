import { useState } from "react"
import { API_URL } from "./api"
import Story from "./Story"
import ElapsedSeconds from "./ElapsedSeconds"

type GenerateStatus = 
  | { status: "none" }
  | { status: "generating", prompt: string }
  | { status: "finished", prompt: string, story: string}
  | { status: "error", error: string }

type GenerateFormProps = {
  onGenerateStart: () => void,
}

const GENERATE_TIMEOUT_MS = 30_000

function renderResult(generateStatus: GenerateStatus) {
  switch(generateStatus.status) {
    case "none":
    case "generating":
      return null
    case "finished":
      return (
        <p className="story story--new">
          <Story story={generateStatus.story} prompt={generateStatus.prompt} />
        </p>
      )
    case "error":
      return (
        <p>Something went wrong.</p>
      )
  }
}

export default function GenerateForm({ onGenerateStart }: GenerateFormProps) {
  // TODO: differentiate error state (rate limited, timout, unknown/other)
  const [promptInput, setPromptInput] = useState("")
  const [generateStatus, setGenerateStatus] = useState<GenerateStatus>({status: "none"})

  function generateStory(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!promptInput.trim()) return

    const promptInputTrimmed = promptInput.trim();
    setGenerateStatus({status: "generating", prompt: promptInputTrimmed})
    onGenerateStart()
    fetch(`${API_URL}/generate`, {
      signal: AbortSignal.timeout(GENERATE_TIMEOUT_MS),
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
        <button disabled={isStoryGenerating}>
          {isStoryGenerating ? <>Generating... <ElapsedSeconds /></> : "Submit"}
        </button>
      </form>
      {renderResult(generateStatus)}
    </>
  )
}
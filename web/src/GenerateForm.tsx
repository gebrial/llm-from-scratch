import { useState } from "react"
import { API_URL } from "./api"
import Story from "./Story"
import ElapsedSeconds from "./ElapsedSeconds"

type GenerateStatus = 
  | { status: "none" }
  | { status: "generating", prompt: string }
  | { status: "finished", prompt: string, story: string}
  | { status: "error", error: GenerateError }

type GenerateError = "rateLimited" | "timeout" | "other"

type GenerateFormProps = {
  onGenerateStart: () => void,
}

const GENERATE_TIMEOUT_MS = 30_000

const ERROR_MESSAGES: Record<GenerateError, string> = {
  rateLimited: "You can make six stories per minute. Try again shortly.",
  timeout: "The story took too long to generate. Try again in a moment.",
  other: "Something went wrong.",
}

class RateLimitedError extends Error {}

function toGenerateError(err: unknown): GenerateError {
  if (err instanceof RateLimitedError) return "rateLimited"
  if (err instanceof DOMException && err.name === "TimeoutError") return "timeout"
  return "other"
}

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
        <p>{ERROR_MESSAGES[generateStatus.error]}</p>
      )
  }
}

export default function GenerateForm({ onGenerateStart }: GenerateFormProps) {
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
      if (res.status === 429) throw new RateLimitedError()
      if (!res.ok) throw new Error(`generate post failed: ${res.status}`)
      return res.json()
    })
    .then((newStory: string) => {
      setGenerateStatus({status: "finished", prompt: promptInputTrimmed, story: newStory})
      setPromptInput("")
    })
    .catch((err: unknown) => {
      console.error(err)
      setGenerateStatus({status: "error", error: toGenerateError(err)})
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
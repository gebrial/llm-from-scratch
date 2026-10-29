import { useState } from "react"
import Story from "./Story"
import ElapsedSeconds from "./ElapsedSeconds"
import { RateLimitedError, streamStory } from "../api/streamStory"

// The continuation is kept on an error, so text that already arrived stays on screen.
type GenerateStatus =
  | { status: "none" }
  | { status: "generating", prompt: string, continuation: string }
  | { status: "finished", prompt: string, continuation: string }
  | { status: "error", error: GenerateError, prompt: string, continuation: string }

type GenerateError = "rateLimited" | "timeout" | "other"

type GenerateFormProps = {
  onGenerateStart: () => void,
}

const ERROR_MESSAGES: Record<GenerateError, string> = {
  rateLimited: "You can make six stories per minute. Try again shortly.",
  timeout: "The story took too long to generate. Try again in a moment.",
  other: "Something went wrong.",
}

function toGenerateError(err: unknown): GenerateError {
  if (err instanceof RateLimitedError) return "rateLimited"
  if (err instanceof DOMException && err.name === "TimeoutError") return "timeout"
  return "other"
}

function renderResult(generateStatus: GenerateStatus) {
  if (generateStatus.status === "none") return null
  const { prompt, continuation } = generateStatus
  return (
    <>
      <p className="story story--new">
        <Story prompt={prompt} continuation={continuation} />
      </p>
      {generateStatus.status === "error" && <p>{ERROR_MESSAGES[generateStatus.error]}</p>}
    </>
  )
}

export default function GenerateForm({ onGenerateStart }: GenerateFormProps) {
  const [promptInput, setPromptInput] = useState("")
  const [generateStatus, setGenerateStatus] = useState<GenerateStatus>({status: "none"})

  function generateStory(event: React.SubmitEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!promptInput.trim()) return

    const prompt = promptInput.trim();
    let continuation = ""
    setGenerateStatus({status: "generating", prompt, continuation})
    onGenerateStart()
    streamStory(prompt, text => {
      continuation += text
      setGenerateStatus({status: "generating", prompt, continuation})
    })
    .then(() => {
      setGenerateStatus({status: "finished", prompt, continuation})
      setPromptInput("")
    })
    .catch((err: unknown) => {
      console.error(err)
      setGenerateStatus({status: "error", error: toGenerateError(err), prompt, continuation})
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
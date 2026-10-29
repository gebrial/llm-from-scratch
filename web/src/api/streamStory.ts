// Reading POST /generate/stream: one JSON message per line (NDJSON), see
// DECISIONS.md, "Stream stories as NDJSON on a new route".

import { API_URL } from "./api"

// One line of the /generate/stream body.
type StreamMessage =
  | { type: "token", text: string }
  | { type: "done" }
  | { type: "error", message: string }

// Reset on every message, so a long story that keeps arriving is never cut off,
// but a stalled one is. Generous enough for the prompt pass on a throttled instance.
const IDLE_TIMEOUT_MS = 15_000

export class RateLimitedError extends Error {}

// Calls onToken with each token's text as it arrives. Resolves on "done", which the
// API sends after the story is stored. Rejects on "error", on a body that ends
// without "done" (the connection broke), or after IDLE_TIMEOUT_MS with no message;
// the last as a DOMException named "TimeoutError", like AbortSignal.timeout.
export async function streamStory(prompt: string, onToken: (text: string) => void): Promise<void> {
  const idleTimeout = startIdleTimeout(IDLE_TIMEOUT_MS)
  try {
    const body = await postForStream(prompt, idleTimeout.signal)
    for await (const line of readLines(body)) {
      idleTimeout.reset()
      const message: StreamMessage = JSON.parse(line)
      if (message.type === "token") onToken(message.text)
      else if (message.type === "done") return
      else throw new Error(`generation failed: ${message.message}`)
    }
    throw new Error("stream ended without done")
  } finally {
    idleTimeout.stop()
  }
}

// An abort signal that fires after `ms` without a reset().
function startIdleTimeout(ms: number) {
  const controller = new AbortController()
  let timer: ReturnType<typeof setTimeout> | undefined
  function reset() {
    clearTimeout(timer)
    timer = setTimeout(() => controller.abort(new DOMException("no data", "TimeoutError")), ms)
  }
  reset()
  return { signal: controller.signal, reset, stop: () => clearTimeout(timer) }
}

async function postForStream(prompt: string, signal: AbortSignal): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(`${API_URL}/generate/stream`, {
    signal,
    method: "POST",
    body: JSON.stringify({ prompt }),
    headers: { "Content-Type": "application/json" }
  })
  if (res.status === 429) throw new RateLimitedError()
  if (!res.ok || !res.body) throw new Error(`generate stream failed: ${res.status}`)
  return res.body
}

// Yields each complete line of the body. A chunk can end mid-line, so the last
// piece of each chunk waits for the next one.
async function* readLines(body: ReadableStream<Uint8Array>): AsyncGenerator<string> {
  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffered = ""
  while (true) {
    const { value, done } = await reader.read()
    if (done) return
    buffered += decoder.decode(value, { stream: true })
    const lines = buffered.split("\n")
    buffered = lines.pop() ?? ""
    yield* lines
  }
}

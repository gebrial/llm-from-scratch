type StoryProps = {
  story: string,
  prompt: string,
}

export default function Story({story, prompt}: StoryProps) {
  // TODO: instead of assuming where to slice the story, compare it with the prompt to compute the continuation
  const continuation = story.slice(prompt.length + 1)
  return (
    <>
      <b className="story-prompt">{prompt}</b>
      {continuation}
    </>
  )
}

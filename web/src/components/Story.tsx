type StoryProps = {
  prompt: string,
  continuation: string,
}

export default function Story({prompt, continuation}: StoryProps) {
  return (
    <>
      <b className="story-prompt">{prompt}</b>
      {continuation}
    </>
  )
}

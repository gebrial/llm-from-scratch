import { FaCode, FaGithub, FaLinkedin } from 'react-icons/fa'
import { MdEmail } from 'react-icons/md'

export default function Footer() {
  return (
    <footer className="app-footer">
      <a href="https://github.com/gebrial/llm-from-scratch" target="_blank" rel="noreferrer" aria-label="Source code" title="Source code"><FaCode /></a>
      <a href="https://github.com/gebrial" target="_blank" rel="noreferrer" aria-label="GitHub" title="GitHub"><FaGithub /></a>
      <a href="https://www.linkedin.com/in/gebrial/" target="_blank" rel="noreferrer" aria-label="LinkedIn" title="LinkedIn"><FaLinkedin /></a>
      <a href="mailto:gebrial@live.ca" aria-label="Email" title="Email"><MdEmail /></a>
    </footer>
  )
}
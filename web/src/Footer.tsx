import { FaBook, FaCode, FaGithub, FaLinkedin } from 'react-icons/fa'
import { MdEmail } from 'react-icons/md'
import { API_URL } from './api'
import type { ServerStatus } from './App'

interface FooterProps {
  serverStatus: ServerStatus
}

export default function Footer({ serverStatus }: FooterProps) {
  return (
    <footer className="app-footer">
      <a href="https://github.com/gebrial/llm-from-scratch" target="_blank" rel="noreferrer" aria-label="Source code" title="Source code"><FaCode /></a>
      {serverStatus === "ready" && (
        <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" aria-label="API docs" title="API docs"><FaBook /></a>
      )}
      <a href="https://github.com/gebrial" target="_blank" rel="noreferrer" aria-label="GitHub" title="GitHub"><FaGithub /></a>
      <a href="https://www.linkedin.com/in/gebrial/" target="_blank" rel="noreferrer" aria-label="LinkedIn" title="LinkedIn"><FaLinkedin /></a>
      <a href="mailto:gebrial@live.ca" aria-label="Email" title="Email"><MdEmail /></a>
    </footer>
  )
}
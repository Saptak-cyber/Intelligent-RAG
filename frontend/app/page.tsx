'use client'

import { useState, useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { marked } from 'marked'

interface Message {
  role: 'user' | 'assistant'
  content: string
}

interface TokenUsage {
  input: number
  output: number
}

interface ResponseMetadata {
  model_used: string
  classification: string
  tokens: TokenUsage
  latency_ms: number
  chunks_retrieved: number
  evaluator_flags: string[]
  evaluator_details?: Array<{
    flag: string
    details: {
      unverified_nouns?: string[]
      response_nouns?: string[]
      chunks_nouns?: string[]
      spacy_available?: boolean
    }
  }>
}

interface Source {
  document: string
  page?: number
  relevance_score?: number
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [conversationId, setConversationId] = useState<string | null>(null)
  const [metadata, setMetadata] = useState<ResponseMetadata | null>(null)
  const [sources, setSources] = useState<Source[]>([])
  const [showDebug, setShowDebug] = useState(true)
  const [useStreaming, setUseStreaming] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const streamingMessageRef = useRef<HTMLDivElement>(null)
  
  // API URL from environment variable or default to localhost
  const API_URL = process.env.NEXT_PUBLIC_API_URL

  // Load chat history from localStorage on mount
  useEffect(() => {
    const savedMessages = localStorage.getItem('chatMessages')
    const savedConversationId = localStorage.getItem('conversationId')
    const savedMetadata = localStorage.getItem('chatMetadata')
    const savedSources = localStorage.getItem('chatSources')

    if (savedMessages) {
      setMessages(JSON.parse(savedMessages))
    }
    if (savedConversationId) {
      setConversationId(savedConversationId)
    }
    if (savedMetadata) {
      setMetadata(JSON.parse(savedMetadata))
    }
    if (savedSources) {
      setSources(JSON.parse(savedSources))
    }
  }, [])

  // Save chat history to localStorage whenever it changes
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem('chatMessages', JSON.stringify(messages))
    }
  }, [messages])

  useEffect(() => {
    if (conversationId) {
      localStorage.setItem('conversationId', conversationId)
    }
  }, [conversationId])

  useEffect(() => {
    if (metadata) {
      localStorage.setItem('chatMetadata', JSON.stringify(metadata))
    }
  }, [metadata])

  useEffect(() => {
    if (sources.length > 0) {
      localStorage.setItem('chatSources', JSON.stringify(sources))
    }
  }, [sources])

  // Auto-scroll to bottom when messages change or loading state changes
  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const clearChat = () => {
    setMessages([])
    setConversationId(null)
    setMetadata(null)
    setSources([])
    localStorage.removeItem('chatMessages')
    localStorage.removeItem('conversationId')
    localStorage.removeItem('chatMetadata')
    localStorage.removeItem('chatSources')
  }

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()
    
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: userMessage }])
    setIsLoading(true)

    if (useStreaming) {
      await handleStreamingSubmit(userMessage)
    } else {
      await handleRegularSubmit(userMessage)
    }
  }

  const handleStreamingSubmit = async (userMessage: string) => {
    let accumulatedText = ''
    let messageIndex = -1
    
    try {
      const response = await fetch(`${API_URL}/query/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage,
          conversation_id: conversationId,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get streaming response')
      }

      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      
      if (!reader) {
        throw new Error('No response body')
      }

      let buffer = ''
      
      // Add empty assistant message
      setMessages(prev => {
        messageIndex = prev.length
        return [...prev, { role: 'assistant', content: '' }]
      })
      
      // Wait for DOM to update
      await new Promise(resolve => setTimeout(resolve, 50))

      while (true) {
        const { done, value } = await reader.read()
        
        if (done) break

        // Decode the chunk and add to buffer
        buffer += decoder.decode(value, { stream: true })
        
        // Process complete lines from buffer
        const lines = buffer.split('\n')
        // Keep the last incomplete line in buffer
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6))
              
              if (data.type === 'token') {
                // Update accumulated text
                accumulatedText += data.content
                console.log('Received token:', data.content, 'Total length:', accumulatedText.length)
                
                // Update DOM directly for immediate visual feedback with markdown rendering
                if (streamingMessageRef.current) {
                  // Convert markdown to HTML and set it
                  streamingMessageRef.current.innerHTML = marked(accumulatedText) as string
                  console.log('Updated DOM ref with markdown')
                  // Force browser to paint the update
                  await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))
                } else {
                  console.log('Ref not available yet')
                }
                
                // Also update state periodically (every 10 tokens to reduce re-renders)
                if (accumulatedText.length % 10 === 0) {
                  setMessages(prev => {
                    const newMessages = [...prev]
                    if (newMessages[messageIndex]) {
                      newMessages[messageIndex] = {
                        role: 'assistant',
                        content: accumulatedText
                      }
                    }
                    return newMessages
                  })
                }
              } else if (data.type === 'metadata') {
                // Update metadata and sources
                if (!conversationId) {
                  setConversationId(data.data.conversation_id)
                }
                setMetadata(data.data.metadata)
                setSources(data.data.sources || [])
              } else if (data.type === 'error') {
                throw new Error(data.error.message)
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError, 'Line:', line)
            }
          }
        }
      }
      
      // Final state update with complete text
      setMessages(prev => {
        const newMessages = [...prev]
        if (newMessages[messageIndex]) {
          newMessages[messageIndex] = {
            role: 'assistant',
            content: accumulatedText
          }
        }
        return newMessages
      })
      
    } catch (error) {
      console.error('Streaming error:', error)
      setMessages(prev => {
        const newMessages = [...prev]
        if (messageIndex >= 0 && newMessages[messageIndex]) {
          newMessages[messageIndex] = {
            role: 'assistant',
            content: accumulatedText || 'Sorry, I encountered an error during streaming. Please try again.'
          }
        }
        return newMessages
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleRegularSubmit = async (userMessage: string) => {
    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: userMessage,
          conversation_id: conversationId,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to get response')
      }

      const data = await response.json()
      
      if (!conversationId) {
        setConversationId(data.conversation_id)
      }

      // Update metadata and sources from response
      setMetadata(data.metadata)
      setSources(data.sources || [])

      setMessages(prev => [...prev, { role: 'assistant', content: data.answer }])
    } catch (error) {
      console.error('Error:', error)
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Sorry, I encountered an error. Please try again.' 
      }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <main className="flex min-h-screen flex-col p-4 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950">
      <div className="w-full max-w-7xl mx-auto flex gap-4 h-[calc(100vh-2rem)]">
        {/* Chat Interface */}
        <div className="flex-1 bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
          {/* Header */}
          <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-5 flex justify-between items-center flex-shrink-0">
            <div>
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <span className="text-3xl">💬</span>
                ClearPath Support
              </h1>
              <p className="text-sm text-blue-100 mt-1">Your intelligent assistant for all things ClearPath</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setUseStreaming(!useStreaming)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all transform hover:scale-105 ${
                  useStreaming 
                    ? 'bg-emerald-500 hover:bg-emerald-600 shadow-lg shadow-emerald-500/50' 
                    : 'bg-slate-700 hover:bg-slate-600'
                }`}
              >
                {useStreaming ? '⚡ Streaming' : '📄 Regular'}
              </button>
              {messages.length > 0 && (
                <button
                  onClick={clearChat}
                  className="px-4 py-2 bg-slate-700 hover:bg-red-600 rounded-lg text-sm font-medium transition-all transform hover:scale-105"
                >
                  🗑️ Clear
                </button>
              )}
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
            {messages.length === 0 ? (
              <div className="text-center text-slate-400 mt-20">
                <div className="text-6xl mb-4"></div>
                <p className="text-xl font-semibold text-slate-300">Welcome to ClearPath Support!</p>
                <p className="text-sm mt-3 text-slate-500">Ask me anything about ClearPath features, pricing, or usage.</p>
                <div className="mt-8 grid grid-cols-1 gap-3 max-w-md mx-auto">
                  {/* <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-left">
                    <p className="text-xs text-slate-400">Try asking:</p>
                    <p className="text-sm text-slate-300 mt-1">"What are the pricing plans?"</p>
                  </div>
                  <div className="bg-slate-800/50 border border-slate-700 rounded-lg p-3 text-left">
                    <p className="text-xs text-slate-400">Try asking:</p>
                    <p className="text-sm text-slate-300 mt-1">"How do I integrate with Slack?"</p>
                  </div> */}
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'} animate-fadeIn`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl p-4 shadow-lg ${
                      message.role === 'user'
                        ? 'bg-gradient-to-br from-blue-600 to-blue-700 text-white'
                        : 'bg-slate-800/80 backdrop-blur-sm border border-slate-700 text-slate-100'
                    }`}
                  >
                    {message.role === 'user' ? (
                      <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                    ) : (
                      <div 
                        className="prose prose-invert prose-sm max-w-none prose-headings:text-slate-200 prose-headings:mt-3 prose-headings:mb-2 prose-p:my-2 prose-p:text-slate-300 prose-ul:my-2 prose-ol:my-2 prose-li:my-1 prose-li:text-slate-300 prose-strong:text-slate-200 prose-code:text-blue-400 prose-code:bg-slate-900/50 prose-code:px-1 prose-code:py-0.5 prose-code:rounded"
                        ref={index === messages.length - 1 ? streamingMessageRef : null}
                      >
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {message.content}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start animate-fadeIn">
                <div className="bg-slate-800/80 backdrop-blur-sm border border-slate-700 rounded-2xl p-4 shadow-lg">
                  <div className="flex space-x-2">
                    <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce"></div>
                    <div className="w-2.5 h-2.5 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2.5 h-2.5 bg-pink-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-slate-800 bg-slate-900/50 backdrop-blur-sm p-4 flex-shrink-0">
            <form onSubmit={handleSubmit} className="flex space-x-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your question... (Shift+Enter for new line)"
                className="flex-1 px-4 py-3 bg-slate-800 border border-slate-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-slate-100 placeholder-slate-500 resize-none transition-all"
                disabled={isLoading}
                rows={1}
                style={{ minHeight: '48px', maxHeight: '120px' }}
                onInput={(e) => {
                  const target = e.target as HTMLTextAreaElement
                  target.style.height = 'auto'
                  target.style.height = Math.min(target.scrollHeight, 120) + 'px'
                }}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-xl hover:from-blue-700 hover:to-purple-700 disabled:from-slate-700 disabled:to-slate-700 disabled:cursor-not-allowed transition-all transform hover:scale-105 disabled:transform-none shadow-lg disabled:shadow-none font-medium self-end"
              >
                {isLoading ? '⏳' : ''} Send
              </button>
            </form>
          </div>
        </div>

        {/* Debug Panel */}
        <div className="w-96 bg-slate-900/50 backdrop-blur-sm border border-slate-800 rounded-2xl shadow-2xl overflow-hidden flex flex-col">
          <div className="bg-gradient-to-r from-slate-800 to-slate-900 text-white p-4 flex justify-between items-center flex-shrink-0 border-b border-slate-700">
            <h2 className="text-lg font-bold flex items-center gap-2">
              <span>🔍</span> Debug Panel
            </h2>
            <button
              onClick={() => setShowDebug(!showDebug)}
              className="text-sm px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors font-medium"
            >
              {showDebug ? '👁️ Hide' : '👁️ Show'}
            </button>
          </div>

          {showDebug && (
            <div className="flex-1 overflow-y-auto p-4 space-y-3 scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">
              {!metadata ? (
                <div className="text-center text-slate-500 mt-20">
                  <div className="text-4xl mb-3">📊</div>
                  <p className="text-sm">Send a message to see debug information</p>
                </div>
              ) : (
                <>
                  {/* Model Used */}
                  <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-3 hover:border-slate-600 transition-colors">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase mb-2 flex items-center gap-1">
                      <span>🤖</span> Model Used
                    </h3>
                    <p className="text-sm font-mono text-blue-400 bg-slate-900/50 px-2 py-1 rounded">{metadata.model_used}</p>
                    <p className="text-xs text-slate-400 mt-2">
                      Classification: <span className="font-semibold text-purple-400">{metadata.classification}</span>
                    </p>
                  </div>

                  {/* Token Usage */}
                  <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-3 hover:border-slate-600 transition-colors">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase mb-2 flex items-center gap-1">
                      <span>🎯</span> Token Usage
                    </h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Input:</span>
                        <span className="text-sm font-semibold text-emerald-400">{metadata.tokens.input}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Output:</span>
                        <span className="text-sm font-semibold text-blue-400">{metadata.tokens.output}</span>
                      </div>
                      <div className="flex justify-between items-center pt-2 border-t border-slate-700">
                        <span className="text-sm font-semibold text-slate-300">Total:</span>
                        <span className="text-sm font-bold text-purple-400">
                          {metadata.tokens.input + metadata.tokens.output}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Evaluator Flags */}
                  <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-3 hover:border-slate-600 transition-colors">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase mb-2 flex items-center gap-1">
                      <span>⚠️</span> Evaluator Flags
                    </h3>
                    {metadata.evaluator_flags.length === 0 ? (
                      <div className="flex items-center space-x-2 bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-2">
                        <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
                        <span className="text-sm text-emerald-400 font-medium">No issues detected</span>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {metadata.evaluator_flags.map((flag, index) => {
                          // Find details for this flag
                          const flagDetails = metadata.evaluator_details?.find(d => d.flag === flag)
                          
                          return (
                            <div key={index} className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-2">
                              <div className="flex items-start space-x-2">
                                <div className="w-2 h-2 bg-yellow-500 rounded-full mt-1.5 animate-pulse"></div>
                                <div className="flex-1">
                                  <p className="text-sm font-semibold text-yellow-400">{flag}</p>
                                  <p className="text-xs text-slate-400 mt-0.5">
                                    {getFlagDescription(flag)}
                                  </p>
                                  
                                  {/* Show unverified nouns if available */}
                                  {flag === 'unverified_feature' && flagDetails?.details?.unverified_nouns && (
                                    <div className="mt-2 pt-2 border-t border-yellow-500/20">
                                      <p className="text-xs font-semibold text-yellow-300 mb-1">Unverified terms:</p>
                                      <div className="flex flex-wrap gap-1">
                                        {flagDetails.details.unverified_nouns.map((noun, i) => (
                                          <span 
                                            key={i}
                                            className="inline-block px-2 py-0.5 bg-yellow-500/20 border border-yellow-500/40 rounded text-xs text-yellow-200"
                                          >
                                            {noun}
                                          </span>
                                        ))}
                                      </div>
                                      {flagDetails.details.spacy_available !== undefined && (
                                        <p className="text-xs text-slate-500 mt-1">
                                          {flagDetails.details.spacy_available ? '✓ Using spaCy NER' : '⚠ Fallback mode (install spaCy)'}
                                        </p>
                                      )}
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    )}
                  </div>

                  {/* Performance Metrics */}
                  <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-3 hover:border-slate-600 transition-colors">
                    <h3 className="text-xs font-semibold text-slate-400 uppercase mb-2 flex items-center gap-1">
                      <span>⚡</span> Performance
                    </h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Latency:</span>
                        <span className="text-sm font-semibold text-cyan-400">{metadata.latency_ms}ms</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-slate-400">Chunks Retrieved:</span>
                        <span className="text-sm font-semibold text-pink-400">{metadata.chunks_retrieved}</span>
                      </div>
                    </div>
                  </div>

                  {/* Sources */}
                  {sources.length > 0 && (
                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-3 hover:border-slate-600 transition-colors">
                      <h3 className="text-xs font-semibold text-slate-400 uppercase mb-2 flex items-center gap-1">
                        <span>📚</span> Sources
                      </h3>
                      <div className="space-y-2">
                        {sources.map((source, index) => (
                          <div key={index} className="text-sm bg-slate-900/50 border border-slate-700 rounded-lg p-2">
                            <p className="font-semibold text-slate-200 truncate">{source.document}</p>
                            <div className="flex justify-between text-xs text-slate-400 mt-1">
                              {source.page && <span>📄 Page {source.page}</span>}
                              {source.relevance_score && (
                                <span className="text-blue-400">⭐ {source.relevance_score.toFixed(3)}</span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </main>
  )
}

function getFlagDescription(flag: string): string {
  const descriptions: Record<string, string> = {
    'no_context': 'Answer generated without relevant documentation',
    'refusal': 'System declined to answer the question',
    'unverified_feature': 'Mentioned features not found in documentation',
    'pricing_uncertainty': 'Pricing information may be uncertain or conflicting'
  }
  return descriptions[flag] || 'Quality warning detected'
}

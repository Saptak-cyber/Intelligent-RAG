'use client'

import { useState, useEffect, useRef } from 'react'

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
  const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

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
                
                // Update DOM directly for immediate visual feedback
                if (streamingMessageRef.current) {
                  streamingMessageRef.current.textContent = accumulatedText
                  console.log('Updated DOM ref')
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
    <main className="flex min-h-screen flex-col p-4 bg-gradient-to-b from-gray-50 to-gray-100">
      <div className="w-full max-w-7xl mx-auto flex gap-4 h-[calc(100vh-2rem)]">
        {/* Chat Interface */}
        <div className="flex-1 bg-white rounded-lg shadow-lg overflow-hidden flex flex-col">
          {/* Header */}
          <div className="bg-blue-600 text-white p-4 flex justify-between items-center flex-shrink-0">
            <div>
              <h1 className="text-2xl font-bold">ClearPath Support</h1>
              <p className="text-sm text-blue-100">Ask me anything about ClearPath</p>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setUseStreaming(!useStreaming)}
                className={`px-4 py-2 rounded-lg text-sm transition-colors ${
                  useStreaming 
                    ? 'bg-green-700 hover:bg-green-800' 
                    : 'bg-blue-900 hover:bg-blue-800'
                }`}
              >
                {useStreaming ? '⚡ Streaming' : '📄 Regular'}
              </button>
              {messages.length > 0 && (
                <button
                  onClick={clearChat}
                  className="px-4 py-2 bg-blue-900 hover:bg-red-800 rounded-lg text-sm transition-colors"
                >
                  Clear Chat
                </button>
              )}
            </div>
          </div>

          {/* Messages Area */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 mt-20">
                <p className="text-lg">Welcome to ClearPath Support!</p>
                <p className="text-sm mt-2">Ask me anything about ClearPath features, pricing, or usage.</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div
                  key={index}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[80%] rounded-lg p-4 ${
                      message.role === 'user'
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-900'
                    }`}
                  >
                    <p 
                      className="whitespace-pre-wrap"
                      ref={index === messages.length - 1 && message.role === 'assistant' ? streamingMessageRef : null}
                    >
                      {message.content}
                    </p>
                  </div>
                </div>
              ))
            )}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg p-4">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="border-t border-gray-200 p-4 flex-shrink-0">
            <form onSubmit={handleSubmit} className="flex space-x-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Type your question... (Shift+Enter for new line)"
                className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-900 resize-none"
                disabled={isLoading}
                rows={1}
                style={{ minHeight: '42px', maxHeight: '120px' }}
                onInput={(e) => {
                  const target = e.target as HTMLTextAreaElement
                  target.style.height = 'auto'
                  target.style.height = Math.min(target.scrollHeight, 120) + 'px'
                }}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors self-end"
              >
                Send
              </button>
            </form>
          </div>
        </div>

        {/* Debug Panel */}
        <div className="w-96 bg-white rounded-lg shadow-lg overflow-hidden flex flex-col">
          <div className="bg-gray-800 text-white p-4 flex justify-between items-center flex-shrink-0">
            <h2 className="text-lg font-bold">Debug Panel</h2>
            <button
              onClick={() => setShowDebug(!showDebug)}
              className="text-sm px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded"
            >
              {showDebug ? 'Hide' : 'Show'}
            </button>
          </div>

          {showDebug && (
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {!metadata ? (
                <div className="text-center text-gray-500 mt-20">
                  <p className="text-sm">Send a message to see debug information</p>
                </div>
              ) : (
                <>
                  {/* Model Used */}
                  <div className="border border-gray-200 rounded-lg p-3">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Model Used</h3>
                    <p className="text-sm font-mono text-gray-900">{metadata.model_used}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      Classification: <span className="font-semibold">{metadata.classification}</span>
                    </p>
                  </div>

                  {/* Token Usage */}
                  <div className="border border-gray-200 rounded-lg p-3">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Token Usage</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Input:</span>
                        <span className="text-sm font-semibold text-gray-900">{metadata.tokens.input}</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Output:</span>
                        <span className="text-sm font-semibold text-gray-900">{metadata.tokens.output}</span>
                      </div>
                      <div className="flex justify-between items-center pt-2 border-t border-gray-200">
                        <span className="text-sm font-semibold text-gray-700">Total:</span>
                        <span className="text-sm font-bold text-gray-900">
                          {metadata.tokens.input + metadata.tokens.output}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Evaluator Flags */}
                  <div className="border border-gray-200 rounded-lg p-3">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Evaluator Flags</h3>
                    {metadata.evaluator_flags.length === 0 ? (
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-sm text-gray-600">No issues detected</span>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {metadata.evaluator_flags.map((flag, index) => (
                          <div key={index} className="flex items-start space-x-2">
                            <div className="w-2 h-2 bg-yellow-500 rounded-full mt-1.5"></div>
                            <div className="flex-1">
                              <p className="text-sm font-semibold text-gray-900">{flag}</p>
                              <p className="text-xs text-gray-500">
                                {getFlagDescription(flag)}
                              </p>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>

                  {/* Performance Metrics */}
                  <div className="border border-gray-200 rounded-lg p-3">
                    <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Performance</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Latency:</span>
                        <span className="text-sm font-semibold text-gray-900">{metadata.latency_ms}ms</span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Chunks Retrieved:</span>
                        <span className="text-sm font-semibold text-gray-900">{metadata.chunks_retrieved}</span>
                      </div>
                    </div>
                  </div>

                  {/* Sources */}
                  {sources.length > 0 && (
                    <div className="border border-gray-200 rounded-lg p-3">
                      <h3 className="text-xs font-semibold text-gray-500 uppercase mb-2">Sources</h3>
                      <div className="space-y-2">
                        {sources.map((source, index) => (
                          <div key={index} className="text-sm">
                            <p className="font-semibold text-gray-900">{source.document}</p>
                            <div className="flex justify-between text-xs text-gray-500">
                              {source.page && <span>Page {source.page}</span>}
                              {source.relevance_score && (
                                <span>Score: {source.relevance_score.toFixed(3)}</span>
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

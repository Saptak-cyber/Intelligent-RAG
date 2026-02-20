# Requirements Document

## Introduction

The ClearPath RAG Chatbot is a customer support system that answers user questions about ClearPath (a fictional SaaS project management tool) by retrieving relevant content from 30 PDF documentation files and generating responses using Large Language Models via the Groq API. The system implements a three-layer architecture: RAG Pipeline for document retrieval, Model Router for intelligent model selection, and Output Evaluator for response quality assessment.

## Glossary

- **RAG_Pipeline**: The Retrieval-Augmented Generation component that processes PDF documents, chunks content, and retrieves relevant passages for query answering
- **Model_Router**: The deterministic rule-based component that classifies queries as simple or complex and routes them to the appropriate LLM
- **Output_Evaluator**: The component that analyzes generated responses and flags potentially unreliable outputs
- **Groq_API**: The API service providing access to LLM models (llama-3.1-8b-instant and llama-3.3-70b-versatile)
- **Document_Chunk**: A strategically segmented portion of a PDF document used for retrieval
- **Relevance_Score**: A numerical value (0-1) indicating how relevant a retrieved chunk is to the user query
- **Classification**: The categorization of a query as either "simple" or "complex" by the Model_Router
- **Evaluator_Flag**: A warning indicator raised by the Output_Evaluator when a response may be unreliable
- **Conversation_Context**: The historical information from previous turns in a multi-turn conversation
- **Query_Endpoint**: The POST /query API endpoint that accepts user questions and returns chatbot responses

## Requirements

### Requirement 1: Document Processing and Chunking

**User Story:** As a system administrator, I want the RAG pipeline to process 30 PDF documentation files and chunk them strategically, so that relevant content can be efficiently retrieved for user queries.

#### Acceptance Criteria

1. THE RAG_Pipeline SHALL load all 30 PDF files from the clearpath_docs/ directory
2. THE RAG_Pipeline SHALL extract text content from each PDF document with page number tracking
3. THE RAG_Pipeline SHALL chunk documents using a strategic segmentation approach without using external RAG libraries
4. THE RAG_Pipeline SHALL preserve document metadata including filename and page numbers for each chunk
5. THE RAG_Pipeline SHALL create a searchable index of all document chunks for retrieval operations

### Requirement 2: Relevant Content Retrieval

**User Story:** As a user, I want the system to retrieve only relevant content for my query, so that I receive accurate answers without unnecessary information.

#### Acceptance Criteria

1. WHEN a user submits a query, THE RAG_Pipeline SHALL retrieve only the most relevant document chunks
2. THE RAG_Pipeline SHALL NOT pass all 30 documents to the LLM for every query
3. WHEN retrieving chunks, THE RAG_Pipeline SHALL compute a relevance score between 0 and 1 for each chunk
4. THE RAG_Pipeline SHALL rank retrieved chunks by relevance score in descending order
5. WHEN no relevant chunks are found, THE RAG_Pipeline SHALL return an empty result set

### Requirement 3: Source Attribution

**User Story:** As a user, I want to see which documents and pages were used to answer my question, so that I can verify the information and explore further.

#### Acceptance Criteria

1. WHEN generating a response, THE System SHALL return source information for each retrieved chunk
2. THE System SHALL include the document filename in each source entry
3. THE System SHALL include the page number in each source entry when available
4. THE System SHALL include the relevance score in each source entry
5. WHEN no chunks are retrieved, THE System SHALL return an empty sources array

### Requirement 4: Deterministic Model Routing

**User Story:** As a system architect, I want queries to be routed to appropriate models using deterministic rules, so that routing decisions are predictable, cost-effective, and not dependent on LLM calls.

#### Acceptance Criteria

1. THE Model_Router SHALL classify every query as either "simple" or "complex" using rule-based logic
2. THE Model_Router SHALL NOT use LLM calls to determine query classification
3. WHEN a query is classified as "simple", THE Model_Router SHALL route it to llama-3.1-8b-instant
4. WHEN a query is classified as "complex", THE Model_Router SHALL route it to llama-3.3-70b-versatile
5. THE Model_Router SHALL produce identical classifications for identical queries (deterministic behavior)
6. THE Model_Router SHALL use explicit signals including query length, keyword presence, number of questions, and intent markers

### Requirement 5: Routing Decision Logging

**User Story:** As a system administrator, I want every routing decision to be logged with detailed metadata, so that I can analyze routing patterns and optimize the system.

#### Acceptance Criteria

1. WHEN the Model_Router classifies a query, THE System SHALL log the complete query text
2. THE System SHALL log the classification result (simple or complex)
3. THE System SHALL log which model was used for generation
4. THE System SHALL log the number of input tokens sent to the LLM
5. THE System SHALL log the number of output tokens generated by the LLM
6. THE System SHALL log the total latency in milliseconds from request to response

### Requirement 6: No-Context Response Detection

**User Story:** As a user, I want to be warned when the chatbot answers without relevant documentation, so that I know the response may not be reliable.

#### Acceptance Criteria

1. WHEN the LLM generates an answer but no relevant chunks were retrieved, THE Output_Evaluator SHALL raise a "no_context" flag
2. THE Output_Evaluator SHALL detect when chunks_retrieved equals zero and the answer is not a refusal
3. WHEN the "no_context" flag is raised, THE System SHALL include it in the evaluator_flags array
4. THE System SHALL still return the generated answer even when the "no_context" flag is raised

### Requirement 7: Refusal Detection

**User Story:** As a user, I want to be notified when the chatbot cannot answer my question, so that I know to seek alternative support channels.

#### Acceptance Criteria

1. WHEN the LLM explicitly refuses to answer or states it doesn't know, THE Output_Evaluator SHALL raise a "refusal" flag
2. THE Output_Evaluator SHALL detect refusal phrases including "I don't have", "not mentioned", and "cannot find"
3. WHEN the "refusal" flag is raised, THE System SHALL include it in the evaluator_flags array
4. THE System SHALL still return the generated answer even when the "refusal" flag is raised

### Requirement 8: Domain-Specific Quality Check

**User Story:** As a product manager, I want the system to detect domain-specific issues in responses, so that users are warned about potentially problematic answers.

#### Acceptance Criteria

1. THE Output_Evaluator SHALL implement at least one domain-specific quality check beyond no_context and refusal detection
2. THE Output_Evaluator SHALL raise a custom flag when the domain-specific condition is detected
3. WHEN the custom flag is raised, THE System SHALL include it in the evaluator_flags array with a descriptive name
4. THE System SHALL still return the generated answer even when the custom flag is raised

### Requirement 9: Query API Endpoint

**User Story:** As a frontend developer, I want a well-defined API endpoint to submit queries and receive responses, so that I can build the chat interface.

#### Acceptance Criteria

1. THE System SHALL expose a POST /query endpoint
2. WHEN receiving a request, THE Query_Endpoint SHALL accept a JSON body with a "question" field
3. THE Query_Endpoint SHALL accept an optional "conversation_id" field for multi-turn conversations
4. WHEN processing a request without a conversation_id, THE System SHALL generate a new conversation identifier
5. THE Query_Endpoint SHALL return a JSON response containing answer, metadata, sources, and conversation_id fields

### Requirement 10: Response Metadata

**User Story:** As a developer, I want detailed metadata about each query processing, so that I can debug issues and monitor system performance.

#### Acceptance Criteria

1. THE System SHALL include a metadata object in every response
2. THE metadata object SHALL contain the model_used field indicating which LLM was invoked
3. THE metadata object SHALL contain the classification field showing the router's decision
4. THE metadata object SHALL contain a tokens object with input and output token counts
5. THE metadata object SHALL contain the latency_ms field showing total processing time
6. THE metadata object SHALL contain the chunks_retrieved field showing how many document chunks were found
7. THE metadata object SHALL contain the evaluator_flags array listing all quality warnings

### Requirement 11: Chat Web Interface

**User Story:** As a user, I want a simple web interface to interact with the chatbot, so that I can ask questions and receive answers easily.

#### Acceptance Criteria

1. THE System SHALL provide a web-based chat interface
2. THE chat interface SHALL allow users to type and submit questions
3. THE chat interface SHALL display the chatbot's answers in a conversational format
4. THE chat interface SHALL be accessible through a web browser
5. THE chat interface SHALL provide a minimal, functional user experience

### Requirement 12: Debug Panel

**User Story:** As a developer, I want a debug panel in the interface showing query processing details, so that I can understand how the system is performing.

#### Acceptance Criteria

1. THE chat interface SHALL include a debug panel displaying query metadata
2. THE debug panel SHALL show which model was used for the current response
3. THE debug panel SHALL show token usage (input and output tokens)
4. THE debug panel SHALL show any evaluator flags raised for the current response
5. THE debug panel SHALL update with each new query and response

### Requirement 13: Groq API Integration

**User Story:** As a system architect, I want the system to use Groq API for LLM inference, so that we can leverage fast, cost-effective model access.

#### Acceptance Criteria

1. THE System SHALL use the Groq API for all LLM inference operations
2. THE System SHALL support the llama-3.1-8b-instant model for simple queries
3. THE System SHALL support the llama-3.3-70b-versatile model for complex queries
4. THE System SHALL authenticate with Groq API using an API key from environment variables
5. THE System SHALL handle Groq API errors gracefully and return appropriate error messages

### Requirement 14: No External RAG Libraries

**User Story:** As a system architect, I want the RAG pipeline implemented without external RAG frameworks, so that we have full control over the retrieval logic and understand the implementation deeply.

#### Acceptance Criteria

1. THE RAG_Pipeline SHALL NOT use LangChain for any functionality
2. THE RAG_Pipeline SHALL NOT use LlamaIndex for any functionality
3. THE RAG_Pipeline SHALL NOT use Gemini File Search or similar managed services
4. THE RAG_Pipeline SHALL implement document chunking using custom logic or general-purpose libraries
5. THE RAG_Pipeline SHALL implement retrieval using custom logic or general-purpose vector search libraries

### Requirement 15: Conversation Memory

**User Story:** As a user, I want the chatbot to remember previous questions in our conversation, so that I can ask follow-up questions naturally.

#### Acceptance Criteria

1. WHEN a conversation_id is provided, THE System SHALL retrieve previous conversation context
2. THE System SHALL include relevant conversation history when generating responses
3. THE System SHALL maintain conversation context across multiple turns
4. WHEN starting a new conversation, THE System SHALL generate a unique conversation_id
5. THE System SHALL store conversation history for retrieval in subsequent turns

### Requirement 16: Streaming Responses

**User Story:** As a user, I want to see the chatbot's response appear progressively, so that I don't have to wait for the complete answer before seeing any output.

#### Acceptance Criteria

1. THE System SHALL support streaming response generation from the LLM
2. WHEN streaming is enabled, THE System SHALL send response tokens as they are generated
3. THE chat interface SHALL display streaming tokens progressively to the user
4. THE System SHALL maintain all metadata tracking during streaming responses
5. THE System SHALL complete the response with full metadata after streaming finishes

### Requirement 17: Evaluation Test Harness

**User Story:** As a quality assurance engineer, I want an automated test harness with predefined queries, so that I can validate system behavior and measure performance.

#### Acceptance Criteria

1. THE System SHALL include a test harness with a collection of predefined test queries
2. THE test harness SHALL execute all test queries and collect responses
3. THE test harness SHALL measure and report routing accuracy for test queries
4. THE test harness SHALL measure and report retrieval quality for test queries
5. THE test harness SHALL generate a summary report of test results

### Requirement 18: Deployment Configuration

**User Story:** As a DevOps engineer, I want clear deployment instructions and configuration options, so that I can deploy the system to various environments.

#### Acceptance Criteria

1. THE System SHALL document all required environment variables in the README
2. THE System SHALL support configuration of the GROQ_API_KEY via environment variable
3. THE System SHALL support configuration of the PORT via environment variable with a default of 8000
4. THE System SHALL provide instructions for local deployment on localhost:8000
5. THE System SHALL be deployable to cloud platforms including AWS, GCP, Vercel, Railway, or Render

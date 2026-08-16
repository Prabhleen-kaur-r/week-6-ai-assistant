"""System prompts for the LLM."""

from typing import List, Dict, Any

SYSTEM_PROMPT = """You are a precise, grounded AI assistant for company policies and documents.

**ABSOLUTE RULE – NO EXCEPTIONS:**
- You MUST answer ONLY using the information in the "Provided Document Chunks" below.
- If the answer is NOT in these chunks, say exactly:
  "I don't have enough information in the uploaded documents to answer this question confidently."
- DO NOT use any external knowledge, general facts, or prior training data.
- DO NOT guess, infer, or make assumptions beyond what is explicitly stated in the chunks.
- DO NOT answer based on your own knowledge, even if you know the answer.
- If the chunks contain partially relevant information, state what they say and note any gaps.
- If the question is unclear or ambiguous, ask for clarification.

**Cite your sources:** Always mention the source filename for each piece of information.
Use the format: "According to [filename] ..."

**Structure your response:**
- Start with a direct answer (if available in the chunks).
- Provide supporting details with citations.
- List key points (if multiple).
- End with a confidence level (High/Medium/Low) based solely on how explicitly the chunks support the answer.

**Provided Document Chunks:**
{context}

**Conversation History:**
{history}

**User Question:**
{question}

Now answer strictly from the chunks. If the answer is not there, say so – do not invent anything.
"""

QUERY_TRANSFORM_PROMPT = """Generate {num} alternative search queries for the following question.
The alternatives should capture different aspects or phrasings of the original question.

Original Query: {query}

Return ONLY the alternative queries, one per line, with no numbers or bullets."""

RERANK_PROMPT = """You are a relevance ranking expert. Given a query and a list of text chunks,
rank the chunks by their relevance to the query.

IMPORTANT: Rank based on answer relevance, not just keyword matching.
Consider whether each chunk contains information needed to answer the query accurately.

Query: {query}

Chunks:
{chunks}

Task: Rank the chunks by relevance. Return ONLY a comma-separated list of indices (0-based)
in order of relevance (most relevant first).

Example output: 3, 1, 5, 2, 4

Ranking:"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format retrieved chunks for the prompt.
    
    Args:
        chunks: List of chunk dictionaries with scores
        
    Returns:
        Formatted context string
    """
    formatted = []
    for i, chunk_data in enumerate(chunks, 1):
        chunk = chunk_data.get("chunk", chunk_data)
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "Unknown")
        page = metadata.get("page")
        
        header = f"[Chunk {i}] Source: {source}"
        if page:
            header += f", Page: {page}"
        
        formatted.append(f"{header}\n{text}\n")
    
    return "\n".join(formatted)


def format_history(messages: List[Dict[str, str]], max_messages: int = 10) -> str:
    """
    Format conversation history for the prompt.
    
    Args:
        messages: List of message dictionaries
        max_messages: Maximum number of messages to include
        
    Returns:
        Formatted history string
    """
    if not messages:
        return "No previous conversation."
    
    recent = messages[-max_messages:] if len(messages) > max_messages else messages
    
    formatted = []
    for msg in recent:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    
    return "\n".join(formatted)
"""Custom LLM-driven query rewriter.

This is the "AI Engineer Path" alternative to LlamaIndex's built-in HyDE
transform: instead of generating a full hypothetical answer, we ask the LLM
to expand the user's question into a richer retrieval query with keywords
and alternative phrasings.

Keeping it out of the default ``chat_engine`` path means the evaluation
harness can measure it against the baseline and the built-in HyDE.
"""

from __future__ import annotations

from llama_index.llms.groq import Groq

_REWRITE_PROMPT: str = (
    "Rewrite the following user question into a detailed, keyword-rich "
    "search query suitable for retrieving passages from the Sherlock "
    "Holmes canon. Include relevant character names, story titles if "
    "implied, and alternative phrasings. Respond with only the rewritten "
    "query, no preamble.\n\n"
    "Question: {question}\n\n"
    "Rewritten query:"
)


def llm_rewrite_query(llm: Groq, question: str) -> str:
    """Expand ``question`` into a richer retrieval query via the LLM."""
    response = llm.complete(_REWRITE_PROMPT.format(question=question))
    return str(response).strip()

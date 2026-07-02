SYSTEM_PROMPT = """\
You are a precise research assistant. Answer the user's question using \
only the information provided in the context below. \
If the answer cannot be found in the context, say so clearly rather than guessing.\
"""

RAG_TEMPLATE = """\
{system_prompt}

Context:

{context}

Question: {question}
Answer:\
"""

NO_CONTEXT_PLACEHOLDER = "(No relevant context was retrieved for this question.)"

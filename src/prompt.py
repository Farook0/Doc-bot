prompt_template = """
Use the following instructions to answer the user's question.
1. If the user greets you (e.g., "hello", "hi", "hey"), respond with a friendly greeting in return.
2. If the information is outside of your knowledge base, respond with a polite statement indicating that you cannot provide an answer and encourage the user to ask further questions.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


import os
from flask import Flask, request, jsonify
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from groq import Groq

app = Flask(__name__)

# Groq setup
GROQ_API_KEY = ""  # Replace with actual key
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize embeddings (required for Chroma to work)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# In-memory conversation history storage (session-wise)
conversation_history = {}

# Chroma DB setup
if os.path.exists("chroma_db") and os.listdir("chroma_db"):
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embedding_function)
else:
    vectorstore = None  # No vector store if it doesn't exist

print("Vectorstore setup complete.")

async def rag(query: str, contexts: list, session_id: str) -> str:
    print("> RAG Called for session:", session_id)

    # Retrieve conversation history for the session
    session_conversation = conversation_history.get(session_id, [])

    context_str = "\n".join(contexts)
    history_str = "\n".join(session_conversation[-5:])  # Keep only last 5 exchanges for context

    prompt = f"""
# Instructions:
- Identify the user's intent.
- If it's a new query, ask clarifying questions.
- If it's an ongoing conversation, acknowledge past responses even if it is a one word response.
- Use only the provided context for answers.
- Keep responses concise and interactive.
- Ask one relevant question at a time.
- For action-related queries (e.g., account creation), ask step-by-step required details.
- Avoid phrases like "I understand". Just reply to the user concisely.
- Do not go beyond document.
- Do not ask repeated questions from user,Just guide them towards the next step.
- if needed respond the user with all the steps involved without skipping any steps.
 -If user ask anything regarding creation then ask question for the reuired field to help them create account and if user asks anything for information then provide the neccessary information from the document.
 


Conversation History:
{history_str}

Context:
{context_str}

User Query: {query}
Bot Response:
"""

    # Generate response using Groq
    completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=1000
    )

    response = completion.choices[0].message.content

    # Store only the last 10 exchanges per session
    conversation_history.setdefault(session_id, []).append(f"User: {query}\nBot: {response}")
    conversation_history[session_id] = conversation_history[session_id][-10:]  # Trim history

    return response

async def retrieve(query: str) -> list:
    print("Retrieving relevant contexts from Chroma DB")
    if vectorstore:
        results = vectorstore.similarity_search(query, k=5)
        contexts = [result.page_content for result in results]
        return contexts
    return []

@app.route("/Rag", methods=['POST'])
async def handle_request():
    data = request.json
    received_message = data.get('payload', '')
    session_id = data.get('session_id', 'default_session')

    print(f"Session: {session_id} | Message: {received_message}")

    contexts = await retrieve(received_message)
    response = await rag(received_message, contexts, session_id)
    
    return jsonify({"message": response})

if __name__ == "__main__":
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=8000)


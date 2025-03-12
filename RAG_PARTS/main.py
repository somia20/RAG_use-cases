
# FIRST MAIN ONLY ONE PATH

import os
import fitz  # PyMuPDF library
from flask import Flask, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

app = Flask(__name__)

# Groq setup
GROQ_API_KEY = ""
groq_client = Groq(api_key=GROQ_API_KEY)

# Initialize embeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# In-memory conversation history storage (stores session-wise history)
conversation_history = {}

# Function to load and process PDF document
def load_and_process_pdf(pdf_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Extract text from all pages
    document_text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        document_text += page.get_text()

    # Split text into chunks using the text splitter
    return text_splitter.create_documents([document_text])

# Initialize or load vectorstore
if os.path.exists("embeddings") and os.listdir("embeddings"):
    vectorstore = Chroma(persist_directory="embeddings", embedding_function=embedding_function)
else:
    # Load PDF file instead of hardcoded text
    pdf_path = "C:/Users/somia.kumari/Downloads/Jawwy AE-6d_Loyalty-Gamification-NBA_HLD_V1.13.pdf"# Update with your PDF file path
    documents = load_and_process_pdf(pdf_path)
    vectorstore = Chroma.from_documents(documents=documents, embedding=embedding_function, persist_directory="embeddings")
    vectorstore.persist()

print(f"Number of documents in vectorstore: {vectorstore._collection.count()}")

async def rag(query: str, contexts: list, session_id: str) -> str:
    print("> RAG Called")
    
    # Retrieve conversation history for the session
    previous_conversations = "\n".join(conversation_history.get(session_id, []))
    
    context_str = "\n".join(contexts)
    prompt = f"""
# When responding:
# - Identify the user's intent.
# - If it's a new query, ask clarifying questions.
# - If it's an ongoing conversation, acknowledge past responses.
# - Only use the provided context for answers.
# - Keep responses concise, break down complex concepts, and suggest next steps.
# -Make the conversation interactive by asking relevant questions at each step,summarizing previous answers and suggesting the possible next step.
# -Do not ask question all at once go step by step.One question at a Time.
# -Do not use phrases like "I understand" and "As per the context".Just ask and reply to user querty in brief and to the point.
# - Ask relevant, open-ended questions to gather more details from the user in order to assist them more effectively.
# - Avoid asking generic or repeated questions, and stay focused on the specific scenario the user is dealing with.
# -If user ask anything regarding creation then ask question for the reuired field to help them create account and if user asks anything for information then provide the neccessary information from the document.
# -If user asks anything regarding creation then first confirm the intent of user by asking question to confirm whether user want to create account or want any information,then ask question for the reuired field to help them create account and if user asks anything for information then provide the neccessary information from the document.

   
    Conversation History:
    {previous_conversations}

    Context:
    {context_str}
    
    Query: {query}
    Answer: """

    # Generate answer using Groq
    completion = groq_client.chat.completions.create(
        messages=[{"role": "system", "content": "You are a helpful assistant."},
                  {"role": "user", "content": prompt}],
        model="llama3-70b-8192",
        temperature=0.2,
        max_tokens=1000
    )

    response = completion.choices[0].message.content

    # Store this conversation in history
    conversation_history.setdefault(session_id, []).append(f"User: {query}\nBot: {response}")
    
    return response

async def retrieve(query: str) -> list:
    print("Retrieving relevant contexts")
    results = vectorstore.similarity_search(query, k=5)
    contexts = [result.page_content for result in results]
    return contexts

@app.route("/")
def hello():
    return "Hello, World!"

async def execute(prompt, session_id):
    print("in execute")
    data = await retrieve(prompt)
    response = await rag(prompt, data, session_id)
    return response

@app.route('/Rag', methods=['POST'])
async def send_sms():
    received_message = request.json.get('payload', '')
    session_id = request.json.get('session_id', 'default_session')  # Default session if not provided
    print("The received message is", received_message)

    s = await execute(prompt=received_message, session_id=session_id)
    return jsonify({"message": s})

if __name__ == "__main__":
    print("in main fn")
    app.run(debug=True, host='0.0.0.0', port=9000)

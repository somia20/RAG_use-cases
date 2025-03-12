from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from groq import Groq  # Updated for Groq
import json
import os

# Set Groq API Key
GROQ_API_KEY = ''  # Replace with your actual Groq API key
groq_client = Groq(api_key=GROQ_API_KEY)

# Load PDF document
pdf_path = 'C:/Users/somia.kumari/Downloads/Jawwy AE-6d_Loyalty-Gamification-NBA_HLD_V1.13.pdf'
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
texts = text_splitter.split_documents(documents)

# Generate embeddings using Sentence Transformer
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to generate 5 QA pairs using LLaMA 70B via Groq
def generate_qa(text):
    prompt = f"""
    Generate 5 detailed question-answer pairs from this text. 
    Make sure each answer is complete and comprehensive, covering all relevant information from the text. 
    Cover all the steps involved for any process if needed. Do not mention any references, images, or section names from the text.
    Also include questions that are related to account creation and Generate all possible questions related to it.

    Requirements for answers:
    1. Include all relevant details and context
    2. Use complete sentences and proper explanations
    3. Cite specific information from the text
    4. Make logical connections between concepts
    5. If needed, answer in points for any steps involved and number the steps like 1,2,3 etc.....
    6. Do not mention any section names, references, or citations.
    7. Include detailed answers if required.

    Text:
    {text}

    Return only in this JSON format:
    [
        {{"question": "Detailed question 1?", "answer": "Comprehensive answer that fully explains the concept..."}},
        {{"question": "Detailed question 2?", "answer": "Complete explanation with all relevant details..."}},
        {{"question": "Detailed question 3?", "answer": "Comprehensive answer covering all key points, with numbered steps like: 1,2,3 etc...."}}
        ...
    ]
    Each answer should be at least 1-2 sentences long and cover the topic thoroughly.
    """

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    # Get the raw response from Groq API
    raw_response = response.choices[0].message.content
    print("Raw Response:", raw_response)

    # Remove any unwanted text or non-JSON content from the beginning of the response
    json_part = raw_response.split('[', 1)[-1]  # Keep everything after the first '['

    # Ensure that we have valid JSON
    try:
        qa_pairs = json.loads('[' + json_part)  # Add an opening bracket back to ensure valid JSON format
        if isinstance(qa_pairs, list):  
            return qa_pairs
        else:
            print("Unexpected response format, skipping...")
            return []
    except json.JSONDecodeError:
        print(f"Error parsing JSON response: {raw_response}")
        return []

# Generate QA pairs from document chunks
all_qa_pairs = []
for text in texts:
    qa_list = generate_qa(text.page_content)
    all_qa_pairs.extend(qa_list)  # Store all QA pairs in a list

# Save QA pairs to a text file
with open('qa_pairs.txt', 'w', encoding='utf-8') as file:
    for qa in all_qa_pairs:
        file.write(f"Q: {qa['question']}\n")
        file.write(f"A: {qa['answer']}\n")
        file.write("\n" + "-"*50 + "\n\n")  # Separator between QA pairs

# Store QA pairs in ChromaDB
persist_directory = "chroma_db"

# Convert QA pairs to a list of text strings for Chroma
formatted_qa_texts = [f"Q: {qa['question']}\nA: {qa['answer']}" for qa in all_qa_pairs]

# Store the formatted QA pairs in ChromaDB
chroma_db = Chroma.from_texts(
    texts=formatted_qa_texts,  # Now stored as proper text
    embedding=embeddings,
    persist_directory=persist_directory
)

# Persist the database
chroma_db.persist()

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from groq import Groq
import json
import os

# Set Groq API Key
GROQ_API_KEY = ''  # Replace with your actual Groq API key
groq_client = Groq(api_key=GROQ_API_KEY)

# Load PDF document
pdf_path = "C:/Users/somia.kumari/Downloads/11_adversarial_search_1_-_minmax.pptx.pdf"
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

# Function to generate 5 MCQs using LLaMA 70B via Groq
def generate_mcq(text):
    prompt = f"""
    Generate 5 detailed multiple-choice questions (MCQs) from this text. 
    Each question should have one correct answer and three distractors (incorrect options).
    Make sure the questions and answers are comprehensive and cover all relevant information from the text.
    Do not mention any references, images, or section names from the text.

    Requirements for MCQs:
    1. Include all relevant details and context in the questions and answers.
    2. Use complete sentences and proper explanations.
    3. Cite specific information from the text.
    4. Make logical connections between concepts.
    5. Do not mention any section names, references, or citations.
    6. Ensure the distractors are plausible but incorrect.
    7. IMPORTANT: The CorrectOption field should contain the FULL TEXT of the correct answer, not just "Option1" etc.

    Text:
    {text}

    Return ONLY in this JSON format without any additional text or explanations:
    {{
      "Question1": {{
        "Question": "Detailed question 1?",
        "Option1": "Option A",
        "Option2": "Option B",
        "Option3": "Option C",
        "Option4": "Option D",
        "CorrectOption": "Option A"
      }},
      "Question2": {{
        "Question": "Detailed question 2?",
        "Option1": "Option A",
        "Option2": "Option B",
        "Option3": "Option C",
        "Option4": "Option D",
        "CorrectOption": "Option B"
      }},
      ...and so on for 5 questions
    }}

    VERY IMPORTANT: For each question, set the "CorrectOption" value to be the FULL TEXT of the correct answer, not just "Option1" or similar. For example, if Option2 is the correct answer and contains "The minimax algorithm", then CorrectOption should be "The minimax algorithm".
    
    Make sure to include exactly 5 questions numbered Question1 through Question5.
    """

    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )

    # Get the raw response from Groq API
    raw_response = response.choices[0].message.content
    print("Raw Response:", raw_response)

    # Try to extract the JSON part
    try:
        # Find the position of the first '{' and the last '}'
        start_pos = raw_response.find('{')
        end_pos = raw_response.rfind('}')
        
        if start_pos != -1 and end_pos != -1:
            json_str = raw_response[start_pos:end_pos+1]
            mcqs = json.loads(json_str)
            
            # Additional validation to ensure CorrectOption contains text, not just option references
            for question_key, question_data in mcqs.items():
                # If CorrectOption is just a reference like "Option1", replace it with the actual text
                if question_data["CorrectOption"].startswith("Option"):
                    option_num = question_data["CorrectOption"].replace("Option", "")
                    if option_num.isdigit() and 1 <= int(option_num) <= 4:
                        option_key = f"Option{option_num}"
                        question_data["CorrectOption"] = question_data[option_key]
            
            if isinstance(mcqs, dict):  
                return mcqs
        
        print("Unexpected response format, skipping...")
        return {}
    except json.JSONDecodeError:
        print(f"Error parsing JSON response: {raw_response}")
        return {}

# Generate MCQs from document chunks
all_mcqs = {}
question_counter = 1

for text in texts:
    chunk_mcqs = generate_mcq(text.page_content)
    if chunk_mcqs:
        for question_key, question_data in chunk_mcqs.items():
            new_key = f"Question{question_counter}"
            all_mcqs[new_key] = question_data
            question_counter += 1

# Save MCQs to a JSON file
with open('mcqs.json', 'w', encoding='utf-8') as file:
    json.dump(all_mcqs, file, indent=4)

# Store MCQs in ChromaDB
persist_directory = "chromaa_db"

# Convert MCQs to a list of text strings for Chroma
formatted_mcq_texts = [
    f"Q: {mcq['Question']}\nOptions: {mcq['Option1']}, {mcq['Option2']}, {mcq['Option3']}, {mcq['Option4']}\nCorrect Answer: {mcq['CorrectOption']}" 
    for mcq in all_mcqs.values()
]

# Store the formatted MCQs in ChromaDB
chroma_db = Chroma.from_texts(
    texts=formatted_mcq_texts,
    embedding=embeddings,
    persist_directory=persist_directory
)

# Persist the database
chroma_db.persist()
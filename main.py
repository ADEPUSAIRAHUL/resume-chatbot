import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# NVIDIA-compatible OpenAI client
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"]
)

# Local vector database
chroma_client = chromadb.PersistentClient(path="./vector_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge"
)

# Local embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def load_resume():
    reader = PdfReader("resume.pdf")

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text
def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

def add_documents(documents):
    embeddings = embedding_model.encode(documents).tolist()

    ids = [
        f"resume-section-{i}"
        for i in range(len(documents))
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )

    print(f"Added {len(documents)} resume sections.")


def search_documents(question, number_of_results=5):
    question_embedding = embedding_model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=number_of_results
    )

    return results["documents"][0]


def chat_with_bot(question):
    documents = search_documents(question, number_of_results=5)

    print("\n[Relevant resume sections found:]")
    for i, document in enumerate(documents, start=1):
        print(f"\n--- Section {i} ---")
        print(document)

    context = "\n\n".join(documents)

    prompt = f"""
Answer the question in detail using only this resume information:

{context}

Question:
{question}

Give a clear, well-structured answer. Do not invent information.
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=4096
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # documents = [
    #     "Our company was founded in 2020.",
    #     "The customer support email is support@example.com.",
    #     "Products can be returned within 30 days.",
    #     "Business hours are Monday to Friday, 9 AM to 5 PM."
    # ]
    resume_text = load_resume()
    documents = split_text(resume_text)

    add_documents(documents)

    print("Chatbot ready. Type 'exit' to stop.")

    while True:
        question = input("You: ")

        if question.lower() in ["exit", "quit", "bye"]:
            break

        try:
            answer = chat_with_bot(question)
            print("Bot:", answer)
        except Exception as error:
            print("Error:", error)

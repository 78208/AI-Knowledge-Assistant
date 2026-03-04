from fastapi import FastAPI, UploadFile, File
import os
import shutil

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

DOCS_FOLDER = "documents"
VECTOR_DB = "vector_db"

os.makedirs(DOCS_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB, exist_ok=True)

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


@app.get("/")
def home():
    return {"message": "AI Knowledge Assistant Running"}


@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/upload-doc")
async def upload_doc(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(DOCS_FOLDER, file.filename)

        # Save PDF
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        # If vector DB already exists → load it
        if os.path.exists(VECTOR_DB + "/index.faiss"):
            db = FAISS.load_local(
                VECTOR_DB,
                embedding,
                allow_dangerous_deserialization=True
            )
            db.add_documents(chunks)

        else:
            db = FAISS.from_documents(chunks, embedding)

        db.save_local(VECTOR_DB)

        return {"message": "Document uploaded and added to vector DB"}

    except Exception as e:
        return {"error": str(e)}
from pydantic import BaseModel

class Question(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(q: Question):
    try:
        # Load vector DB
        db = FAISS.load_local(
            VECTOR_DB,
            embedding,
            allow_dangerous_deserialization=True
        )

        # Search similar text
        docs = db.similarity_search(q.question, k=3)

        if not docs:
            return {"answer": "Information not found in the documents."}

        context = "\n".join([doc.page_content for doc in docs])

        return {
            "question": q.question,
            "answer": context
        }

    except Exception as e:
        return {"error": str(e)}
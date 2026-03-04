# AI Knowledge Assistant

This project is an AI Knowledge Assistant built using FastAPI, LangChain, HuggingFace embeddings, and FAISS vector database.

The system allows users to upload PDF documents and ask questions based on the content of those documents.

## Features
- Upload PDF documents
- Text chunking using LangChain
- Embeddings using HuggingFace
- Vector search using FAISS
- Question answering from documents

## API Endpoints

### Upload Document
POST /upload-doc

### Ask Question
POST /ask

### Health Check
GET /health

## Technologies Used
- Python
- FastAPI
- LangChain
- HuggingFace Embeddings
- FAISS

## Run the Project

Start server:

uvicorn app.main:app --reload

Open Swagger UI:

http://127.0.0.1:8000/docs

## GitHub Repository
https://github.com/78208/AI-Knowledge-Assistant

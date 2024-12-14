# PDF RAG System

A FastAPI-based PDF Retrieval-Augmented Generation system that provides an intuitive web interface for document processing and intelligent querying.

## 🌟 Overview

This system combines vision models for accurate text extraction with language models for intelligent document querying, all wrapped in a user-friendly web interface.

## ⚙️ System Components

- **Backend Framework**: FastAPI
- **Vector Storage**: ChromaDB
- **LLM Integration**: Ollama
- **PDF Processing**: PyMuPDF
- **Frontend Styling**: TailwindCSS (via CDN)

## 🚀 Features

- Upload and process multiple PDF files simultaneously
- Extract text using vision models for superior accuracy
- Store and index document embeddings for fast retrieval
- Natural language querying of document content
- Download document transcriptions as JSON
- View answer sources and references
- Clean web interface with real-time status updates

## 📋 Prerequisites

### Required Python Packages

```bash
pip install fastapi uvicorn chromadb ollama PyMuPDF Pillow python-multipart
```

### Required Services
1. **Ollama** must be running locally on port 11434
2. Pull these models using Ollama:
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
ollama pull llama3.2-vision
```

## 🛠️ Setup & Running

1. Clone the repository
2. Create a `static` directory in the project root:
```bash
mkdir static
```
3. Start the server:
```bash
python app.py
```
4. Access the web interface at: `http://localhost:8005`

## 💻 Web Interface

The interface provides:
- PDF upload section with multi-file support
- Transcription download functionality
- Question input for document querying
- Response display with source references
- Real-time status updates for all operations

## 🔄 Workflow

1. **Upload**: Submit one or more PDF files
2. **Processing**: 
   - Vision model extracts text
   - Text is embedded and stored in ChromaDB
3. **Querying**:
   - Enter natural language questions
   - System retrieves relevant context
   - LLM generates precise answers with sources

## 🔒 Security Note

This application is configured for local use. Implement appropriate security measures before deploying in a production environment.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.
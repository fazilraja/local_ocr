from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List
import chromadb
from ollama import Client
import fitz  
from PIL import Image
import datetime
import os
import json

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize clients
OLLAMA_HOST = "http://localhost:11434"  # Changed from 0.0.0.0 to localhost
ollama_client = Client(host=OLLAMA_HOST)
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="pdf_docs")

# models
TEXT_MODEL = "llama3.2"
EMBEDDING_MODEL = "mxbai-embed-large"
VISION_MODEL = "llama3.2-vision"

APP_HOST = "localhost"
APP_PORT = 8005

#frontend
@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF RAG System</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 p-8">
        <div class="max-w-4xl mx-auto">
            <h1 class="text-3xl font-bold mb-8">PDF RAG System</h1>
            
            <div class="bg-white p-6 rounded-lg shadow-md mb-8">
                <h2 class="text-xl font-semibold mb-4">Upload PDFs</h2>
                <form id="uploadForm">
                    <input type="file" multiple accept=".pdf" class="mb-4 p-2 w-full border rounded"/>
                    <div class="flex gap-4">
                        <button type="submit" class="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600">
                            Upload
                        </button>
                        <button type="button" onclick="downloadTranscription()" class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                            Download Transcription
                        </button>
                    </div>
                </form>
                <div id="uploadStatus" class="mt-4 text-sm"></div>
            </div>
            <div class="bg-white p-6 rounded-lg shadow-md">
                <h2 class="text-xl font-semibold mb-4">Ask Questions</h2>
                <textarea id="queryInput" rows="3" class="w-full p-2 border rounded mb-4" 
                          placeholder="Enter your question..."></textarea>
                <button onclick="askQuestion()" class="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600">
                    Ask
                </button>
                <div id="response" class="mt-4 p-4 bg-gray-50 rounded hidden"></div>
            </div>
        </div>
        <script>
            // Function to download transcription
            async function downloadTranscription() {
                try {
                    const response = await fetch('/static/latest_transcription.json');
                    const data = await response.json();
                    
                    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'transcription.json';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                } catch (error) {
                    alert('No transcription available for download');
                }
            }
            document.getElementById('uploadForm').onsubmit = async (e) => {
                e.preventDefault();
                const files = e.target.querySelector('input[type="file"]').files;
                const status = document.getElementById('uploadStatus');
                
                if (files.length === 0) return;
                const formData = new FormData();
                for (let file of files) {
                    formData.append('files', file);
                }
                status.textContent = 'Uploading and processing...';
                status.className = 'mt-4 text-sm text-blue-500';
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        status.textContent = 'Files processed successfully!';
                        status.className = 'mt-4 text-sm text-green-500';
                    } else {
                        throw new Error(result.detail || 'Upload failed');
                    }
                } catch (error) {
                    status.textContent = `Error: ${error.message}`;
                    status.className = 'mt-4 text-sm text-red-500';
                }
            };
            async function askQuestion() {
                const query = document.getElementById('queryInput').value;
                const responseDiv = document.getElementById('response');
                
                if (!query.trim()) return;
                responseDiv.textContent = 'Thinking...';
                responseDiv.className = 'mt-4 p-4 bg-gray-50 rounded';
                responseDiv.style.display = 'block';
                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ query })
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        responseDiv.innerHTML = `
                            <div class="font-medium">Answer:</div>
                            <div class="mt-2">${result.response}</div>
                            ${result.sources ? `
                                <div class="mt-4 text-sm text-gray-500">
                                    <div class="font-medium">Sources:</div>
                                    <div>${result.sources.join(', ')}</div>
                                </div>
                            ` : ''}
                        `;
                    } else {
                        throw new Error(result.detail || 'Query failed');
                    }
                } catch (error) {
                    responseDiv.textContent = `Error: ${error.message}`;
                    responseDiv.className = 'mt-4 p-4 bg-red-50 rounded text-red-500';
                }
            }
        </script>
    </body>
    </html>
    """

def process_pdf_to_text(pdf_bytes: bytes) -> List[str]:
    texts = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        img_filename = f"page_{page_num}_{timestamp}.png"
        img_path = os.path.join(STATIC_DIR, img_filename)
        img_data.save(img_path)
            
        response = ollama_client.chat(
            model=VISION_MODEL,
            messages=[{
                'role': 'user',
                "content": "Your task is to transcribe all visible text from the provided image with accuracy and completeness. Follow these guidelines:\n\n1. **Text Extraction**: Identify and transcribe all text present in the image, including printed, handwritten, or stylized text, while preserving the structure and order in which it appears.\n\n2. **Formatting Consistency**: Maintain the original formatting, such as line breaks, bullet points, or numbered lists, where applicable.\n\n3. **Clarity**: Ensure the transcription is clear and does not contain errors or omissions.\n\n4. **Handle Unclear Text**: For text that is unclear, illegible, or partially obscured, indicate this explicitly using placeholders (e.g., '[illegible]').\n\n5. **Exclude Non-Text Elements**: Focus solely on text; do not describe images, graphics, or non-textual elements unless explicitly requested.\n\n6. **Professional Tone**: Ensure the transcription is neutral and professionally formatted, ready for further use or review.",
                'images': [img_path]
            }]
        )
        texts.append(response['message']['content'])
        
    doc.close()
    return texts

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using the EMBEDDING_MODEL."""
    response = ollama_client.embeddings(model=EMBEDDING_MODEL, prompt=text)
    return response["embedding"]

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        all_texts = []
        for file in files:
            content = await file.read()
            texts = process_pdf_to_text(content)
            all_texts.append({
                "filename": file.filename,
                "pages": texts
            })
            
            embeddings = [generate_embedding(text) for text in texts]
            ids = [f"{file.filename}_page_{i}" for i in range(len(texts))]
            
            collection.add(
                embeddings=embeddings,
                documents=texts,
                ids=ids,
                metadatas=[{"filename": file.filename}] * len(texts)
            )
        
        # Save transcription to JSON file
        json_path = os.path.join(STATIC_DIR, "latest_transcription.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_texts, f, ensure_ascii=False, indent=2)
        
        return {"message": "Files processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: dict):
    try:
        print("\n=== DEBUG: Starting query processing ===")
        print(f"Received query: {query['query']}")

        # Debug embedding generation
        query_embedding = generate_embedding(query["query"])
        print(f"Generated embedding length: {len(query_embedding)}")

        # Debug ChromaDB query
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
            include=["documents", "metadatas"]
        )
        print("\nChromaDB Results:")
        print(f"Number of documents found: {len(results['documents'][0])}")
        print(f"Document IDs: {results['ids']}")
        print(f"Metadatas: {results['metadatas']}")

        # Debug context creation
        context = "\n".join(results["documents"][0])
        print(f"\nContext length: {len(context)}")
        print(f"First 200 chars of context: {context[:200]}...")

        # Debug LLM call
        print("\nSending to LLM with messages:")
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant answering questions about a resume. Use only the information from the resume to answer questions."
            },
            {
                "role": "user",
                "content": f"Resume content: {context}\n\nQuestion: {query['query']}"
            }
        ]
        print(f"Messages: {messages}")

        response = ollama_client.chat(
            model=TEXT_MODEL,
            messages=messages
        )
        print(f"\nLLM Response: {response['message']['content']}")

        return {
            "response": response['message']['content'],
            "sources": list(set(m.get("filename", "Unknown") for m in results["metadatas"][0]))
        }
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_HOST, port=APP_PORT)
# **AI Client Email Assistant — Retrieval-Augmented (RAG) Email Generator (BERT + GPT + FAISS + Flask)**




https://github.com/user-attachments/assets/cc6e4888-2d25-4470-879a-2cf2d207fd1f


---

## **Overview**

This project implements a full **Retrieval-Augmented Email Generator (RAG)** using:

- **Sentence Transformers (MiniLM)** for dense embeddings  
- **Chunking + FAISS** for fast vector search  
- **PDF/TXT ingestion** with page-aware extraction  
- **GPT-2** for generating accurate client email drafts  
- **Flask** for a clean, simple UI  

Users can upload documents, enter purpose/tone/client type/key-points, and receive multiple **context-grounded** email variants.

---

## **High-Level Architecture**

### **1. Document Ingestion (PDF/TXT)**  
- PDF pages extracted using PyPDF2  
- Each page tagged as:  
- TXT files read directly  
- User’s “past conversation” text merged with uploaded docs  

---

### **2. Chunking Layer**  
Long documents are automatically chunked (~300 characters each):

- Preserves semantic structure  
- Ensures better retrieval  
- Prevents model confusion from long contexts  


---

### **3. Vector Indexing with FAISS**

Embeddings of all chunks are generated using:

**Model:** `all-MiniLM-L6-v2`

Then stored in a **FAISS L2 index** for fast retrieval.

- Supports efficient top-k search  
- Rebuilt only when new documents are uploaded  
- Replaces slow cosine similarity loops  

---

### **4. Retrieval Step**

A retrieval query is constructed using:


FAISS returns the **top-k most relevant chunks**, pulled from:

- Uploaded documents  
- Past email conversation text  

These chunks become GPT-2’s grounding context.

---

### **5. Prompt Construction**

Structured template:


This enforces clean, reliable generation.

---

### **6. Controlled Email Generation (GPT-2)**  
Model: **gpt2-medium**

Controls applied:
- `temperature = 0.1`  
- `no_repeat_ngram_size = 2`  
- `top_k = 50`  
- Token budget aware generation (`max_new_tokens <= 1024`)  

Outputs:
- Multiple polished email variants  
- Downloadable as `.txt` files  

---

### **7. Flask Web Interface**

Files:
- `index.html` — Inputs (purpose, tone, client type, key points, PDF/TXT upload)  
- `result.html` — Generated email variants  
- `style.css` — UI styling  

Supports:
- Multiple file uploads  
- Custom retrieval parameters  
- Custom token limits  
- Clean layout for easy use  

---

## **Project Structure**

main_logic.py  # Retrieval, chunking, FAISS, GPT-2 generation<br>

app.py # Flask server and ingestion pipeline <br>

templates/ <br>
  -- index.html <br>
  -- result.html<br>

static/<br>
  -- style.css<br>
  -- README.md 

---

## **How to Run Locally**

### **1. Install Dependencies**
  - pip install -r requirements.txt
  - You may need:
  - pip install faiss-cpu PyPDF2 sentence-transformers transformers torch Flask

### **2. Start the Flask Server**
  - python app.py
  
### **3. Open in Browser**
  - http://127.0.0.1:5000/

## **Key Learnings **
- Retrieval is more important than model size

- Small models + RAG = powerful, accurate automation

- Chunking + FAISS dramatically improves speed

- Page-aware extraction improves business context understanding

- Structured prompts yield consistent email output

--- 

## **Community Question**
### **Which enhancement would you pick next?**

- Reranking with cross-encoders
- Overlapping chunking
- Chat memory
- Upgrading GPT-2 to a modern LLM

<br>

#NLP #RAG #BERT #GPT2 #FAISS #EmailAutomation #Python #Flask #AI #DeepLearning #MachineLearning


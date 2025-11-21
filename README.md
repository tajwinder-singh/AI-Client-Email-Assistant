# AI Client Email Assistant — Retrieval + GPT Generator



https://github.com/user-attachments/assets/8dafc8a1-1a31-4575-8b9e-813c7d26e3f8



## Overview
This project implements a **client email generation assistant** that combines:
- **Sentence Transformers (MiniLM)** for retrieving context from past emails
- **GPT-2** for generating refined, context-aware email variants
- **Flask** for a clean and interactive web interface

The user inputs the purpose, tone, client type, key points, and past conversation.  
The system retrieves relevant past emails and generates multiple polished email drafts.

---

## System Architecture

### 1. Retrieval (Sentence Transformers)
- Model: `all-MiniLM-L6-v2`  
- Encodes user purpose + key points (`query_for_retrieval` in app.py :contentReference[oaicite:5]{index=5}).  
- Encodes all past emails.  
- Computes cosine similarity using `util.cos_sim`.  
- Retrieves the **top-k** past emails (retrieve_context() in main_logic.py :contentReference[oaicite:6]{index=6}).

The retrieved emails become the grounding context for GPT-2.

---

### 2. Prompt Construction
Using a structured template (PROMPT_TEMPLATE in main_logic.py :contentReference[oaicite:7]{index=7}):



This ensures GPT-2 receives clean, organized guidance.

---

### 3. Controlled Email Generation (GPT-2)
- Model: `gpt2-medium`  
- Temperature: **0.1** (stable, precise output)  
- Top-k: **50**  
- No-repeat-ngram: **2**  
- Max new tokens: capped at **1024 – context length** (handled in main_logic.py).

The function `generate_email_gpt()` outputs multiple cleaned variants.  
(app.py passes them to result.html for display :contentReference[oaicite:8]{index=8})

---

### 4. Web Interface (Flask + HTML/CSS)
**index.html** :contentReference[oaicite:9]{index=9}  
- Form inputs for purpose, client type, tone, key points, past emails, top-k, token limits.  
- Clean UI built using `style.css` :contentReference[oaicite:10]{index=10}.

**result.html** :contentReference[oaicite:11]{index=11}  
- Displays prompt and generated email variants.  
- Each variant can be downloaded via `/download`.

---

## Learnings
- Retrieval improves generation reliability by grounding the model in real history.  
- Structured prompts produce more consistent outputs than free-form prompts.  
- Separation of retrieval + generation keeps the system modular and easy to debug.

---

## Tech Stack
- Python  
- Flask  
- Sentence Transformers  
- GPT-2 (HuggingFace)  
- HTML, CSS, Jinja2  
- Torch  

---

## How to Run
1. Install dependencies:

2. Run the server:


3. Open in browser:
https://127.0.0.1:5000/


---

## Question for Readers
Do you think client-facing email tools should focus more on  
**retrieval accuracy** or **generation quality**?

---

#NLP #EmailAutomation #SentenceTransformers #GPT2 #Flask #RAG #AI #Python

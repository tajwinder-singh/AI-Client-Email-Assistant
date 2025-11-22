from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer
import textwrap
from typing import List # This is used to specify the data type of list in the function's parameter.
import torch
import re

# Libraries for FAISS
try: 
    import faiss
    import numpy as np
except Exception:
    faiss=None
    np = None

# =========================
# PROMPT TEMPLATE
# ========================
# '.dedent' wraps (formats) the text with the equal indentation (equal space to the right-end of text) of the prompt.
PROMPT_TEMPLATE = textwrap.dedent("""
Write a {tone} email to a {client_type}.
Purpose: {purpose}
Past Conversation: {past}
Key points: {key_points}

Email:
""")


# =============================
# MODEL LOADING (TO BE CALLED FROM Flask)
# =============================
def load_bert_model(model_name: str = "all-MiniLM-L6-v2"): 
    return SentenceTransformer(model_name)

def load_gpt_model(model_name: str = "gpt2-medium"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval() 
    return tokenizer, model


# ===============================
# CHUNKING UTIL
# ===============================
def chunk_text(text: str, chunk_size: int=300, overlap: int = 50):
    """
    Splits large documents / emails into chunks of ~300 tokens.
    """
    if not text or not text.strip():
        return [] # If text is empty, immediately returns and empty list since the current function returns a list.
    
    sentences = re.split(r'(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<!\bSr)(?<!\bJr)(?<!\bSt)(?<=[.!?])\s+' text.strip()) 
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current.strip()) # Save completed chunk.
            current = s.strip() # Start new chunk with current sentence.

    if current:
        chunks.append(current.strip())

    if overlap > 0 and len(chunks) > 1: # Overlap helps to keep the last last 50 words of the previous chunk into the current chunk so that BERT captures the correct context. This give the context of the of the previous chunk which helps GPT to understand that the current chunk is the continuation of the previous chunk. If overlap > 0 and len(chunks) > 1 then only there is a need to do this.
        overlapped_chunks = [] 
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk) # appending the first chunk as it is.
            else:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk 
                combined = (overlap_text + " " + chunk).strip()
                overlapped_chunks.append(combined)
        
        return overlapped_chunks 
     
    return chunks 

# ==================================
# FAISS HELPERS
# ==================================
def build_faiss_index(bert_model, chunks):
    if faiss is None:
        raise ImportError("FAISS is not installed. Please install faiss-cpu.")

    vectors = bert_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors) 

    return index, vectors


# ==================================
# RETRIEVAL LOGIC
# ==================================
def retrieve_context(bert_model: SentenceTransformer, past_emails: List[str], query: str, top_k: int=2): 
    """
    Encode the past emails, query and return the context of top past emails based on top-k.
    """
    if not past_emails:
        return ""

    all_chunks = []
    for email in past_emails:
        all_chunks.extend(chunk_text(email))

    if len(all_chunks) == 0:
        return ""
    

    faiss_index, vectors = build_faiss_index(bert_model, all_chunks)
    q_vec = bert_model.encode([query], convert_to_numpy=True).astype("float32") # FAISS expects vector to be float32.
    
    top_k = min(top_k, len(all_chunks)) 
    distances, indices = faiss_index.search(q_vec, top_k) # Now, applying search inside 'faiss_index' to get the top_k indices.
    context_list = [all_chunks[i] for i in indices[0]]   
    
    return "\n---\n".join(context_list) 


# ==================================
# GENERATION LOGIC
# =================================
def generate_email_gpt(tokenizer, model, prompt: str, max_new_tokens: int = 220):
    if not prompt:
        return [""] 
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids # shape: (batch=1, seq_len)
    attention_mask = inputs.get("attention_mask", None) # If 'inputs' has attention_mask, then return attention_mask else return 'None'.

        
    max_model_tokens = 1024
    context_token_allowance = max(max_model_tokens - max_new_tokens, 1)
    seq_len = input_ids.shape[1] # capturing the seq_len
    if seq_len > context_token_allowance:
        input_ids = input_ids[:, -context_token_allowance:] 
        if attention_mask is not None: 
            attention_mask = attention_mask[:, -context_token_allowance:]
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    with torch.no_grad():
        outputs = model.generate(
            input_ids, 
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=0.1, 
            do_sample=True, 
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=50,
            no_repeat_ngram_size=2
        
        )
        
    decoded = [tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    
    # strip prompt prefix if present and return list of cleaned string(s)
    cleaned = []
    for r in decoded:
        if r.startswith(prompt):
            r_text=r[len(prompt):].strip()
        else:
            r_text=r.strip()
        cleaned.append(r_text)
    return cleaned


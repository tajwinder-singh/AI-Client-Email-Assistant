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
def load_bert_model(model_name: str = "all-MiniLM-L6-v2"): # "str = 'all-MiniLM-L6-v2'" specifies that the model name should be string (doesn't a strict data-type-requirement, rather it is a demonstration for the coder). If the user doesn' specifiy the model name in the argument (during function call), then the default model will be 'all-MiniLM-L6-v2' only.
    return SentenceTransformer(model_name)

def load_gpt_model(model_name: str = "gpt2-medium"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval() # Model will automatically come into evaluation state once loaded. It was optional, but it is recommended in production.
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
    
    sentences = re.split(r'(?<=[.!?])\s+', text.strip()) # '(?<=[.!?])\s+' splits sentences on '.', '!' and '?'. '\s+' removes the whitespaces coming after one of these three punctuations, and then splits the sentences without preserving the an empty space at the start of the sentence. That's why we use " " + s in current = (...) to give a space at the start of every sentence except the first one. The remaining patters are used to avoid the split when there is some sentence in the text: "Dr. Raj met Prof. Arora. They discussed the project!", we don't want the 'Dr.' and 'Raj met Prof.' to be separate sentences instead we want: ["Dr. Raj met Prof. Arora.", "They discussed the project!"] that's why we use the other patterns. 
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
        overlapped_chunks = [] # Stores the overlapped_chunks which we talked about.
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk) # appending the first chunk as it is.
            else:
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk # only overlap when len is > 50.
                combined = (overlap_text + " " + chunk).strip()
                overlapped_chunks.append(combined)
        
        return overlapped_chunks # if the 'if' condition satisfies, returns the overlapped_chunks instead of returning 'chunks'.
     
    return chunks 

# ==================================
# FAISS HELPERS
# ==================================
# Facebook AI Similarity Search (FAISS) is a library that helps to search the similar vector embeddings in a faster way without recomputing them. It creates a temporary memory internally. We don't need to compute cos.sim instead we use FAISS for doing the same job.
def build_faiss_index(bert_model, chunks):
    if faiss is None:
        raise ImportError("FAISS is not installed. Please install faiss-cpu.")

    vectors = bert_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False).astype("float32")

    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim) # IndexFlatL2 uses euclidean distance formula internally. It takes the length of embeddings.
    index.add(vectors) # vectors is the doc embeddings. Later, when called inside retrieve_context(), it will take the query and search the similar docs and returns them on the basis of top_k indices.

    return index, vectors


# ==================================
# RETRIEVAL LOGIC
# ==================================
def retrieve_context(bert_model: SentenceTransformer, past_emails: List[str], query: str, top_k: int=2): # "bert_model: SentenceTransformer" specifies that the bert_model should be a variable that holds a class 'SentenceTransformer'. It does not strictly required, but is useful for the coder to show that bert_model should be/is a SentenceTransformer.
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
    
    top_k = min(top_k, len(all_chunks)) # top_k=min(top_k, len(all_chunks))' is required to avoid error as if the top_k > len(all_chunks), then the out of index error will occur. So, top_k should be <= len(all_chunks).
    distances, indices = faiss_index.search(q_vec, top_k) # Now, applying search inside 'faiss_index' to get the top_k indices.
    context_list = [all_chunks[i] for i in indices[0]]   
    
    return "\n---\n".join(context_list) # "\n---\n" separates each email in a better way so that GPT2 can understand where the next email starts and end.


# ==================================
# GENERATION LOGIC
# =================================
def generate_email_gpt(tokenizer, model, prompt: str, max_new_tokens: int = 220):
    if not prompt:
        return [""] # The remaining code will not executed and this returns empty string in the list for this entire function instead of returning 'cleaned'.
    
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids # shape: (batch=1, seq_len)
    attention_mask = inputs.get("attention_mask", None) # If 'inputs' has attention_mask, then return attention_mask else return 'None'.

        
    max_model_tokens = 1024
    context_token_allowance = max(max_model_tokens - max_new_tokens, 1)
    seq_len = input_ids.shape[1] # capturing the seq_len
    if seq_len > context_token_allowance:
        input_ids = input_ids[:, -context_token_allowance:] # Since input_id has shape:(n, seq_len), thats why indexing at dimension 1 to get last tokens.
        if attention_mask is not None: # 'inputs' from gpt_tokenizer produces an attention mask of the same shape of 'input_ids'. If we are truncating the input_ids to get the last tokens, then we need to also truncate the attention mask since both we produced at the same time.
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

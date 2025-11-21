from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util
import textwrap
from typing import List # This is used to specify the data type of list in the function's parameter.
import torch

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
# MODEL LOADING (TO BE CALLED FROM STREAMLIT)
# =============================
def load_bert_model(model_name: str = "all-MiniLM-L6-v2"): # "str = 'all-MiniLM-L6-v2'" specifies that the model name should be string (doesn't a strict data-type-requirement, rather it is a demonstration for the coder). If the user doesn' specifiy the model name in the argument (during function call), then the default model will be 'all-MiniLM-L6-v2' only.
    return SentenceTransformer(model_name)

def load_gpt_model(model_name: str = "gpt2-medium"):
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval() # Model will automatically come into evaluation state once loaded. It was optional, but it is recommended in production.
    return tokenizer, model



# ==================================
# RETRIEVAL LOGIC
# ==================================
def retrieve_context(bert_model: SentenceTransformer, past_emails: List[str], query: str, top_k: int=2): # "bert_model: SentenceTransformer" specifies that the bert_model should be a variable that holds a class 'SentenceTransformer'. It does not strictly required, but is useful for the coder to show that bert_model should be/is a SentenceTransformer.
    """
    Encode the past emails, query and return the context of top past emails based on top-k.
    """
    if not past_emails:
        return ""

    emb_docs = bert_model.encode(past_emails, convert_to_tensor = True, show_progress_bar=False)
    q_emb = bert_model.encode(query, convert_to_tensor=True, show_progress_bar=False)
    sim_scores = util.cos_sim(q_emb, emb_docs)
    top_indices = torch.topk(sim_scores, k=min(top_k, len(past_emails))).indices[0].tolist() # 'k=min(top_k, len(past_emails))' is required to avoid error as if the top_k > len(past_emails), then the out of index error will occur. So, top_k should be <= len(past_emails). 'torch.topk' returns two seperate tupels of names: (values, indices), each tuple is of the shape: (num_of_queries, num_of_embeddings), so to get the indices tuple and the num_of_embeddings (which are present in the dim = 1) we use 'indices[0]'
    context_list = [past_emails[i] for i in top_indices]
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
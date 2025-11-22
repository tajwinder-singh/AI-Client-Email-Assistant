from flask import Flask, render_template, request, send_file
import tempfile
import os
import re


import torch
from main_logic import (
    load_bert_model, 
    load_gpt_model,
    retrieve_context,
    generate_email_gpt,
    PROMPT_TEMPLATE
)


app = Flask(__name__)


# --------------------------------
# LOADING MODELS ONCE
# --------------------------------
bert = load_bert_model("all-MiniLM-L6-v2")
tokenizer, gpt_model = load_gpt_model("gpt2-medium")


# --------------------------------
# HOME PAGE (FORM)
# --------------------------------
@app.route("/", methods=["GET"]) # 'index.html' will shows the web page interface.
def index():
    return render_template("index.html")


# --------------------------------
# PROCESS FORM AND GENERATE EMAIL
# --------------------------------
@app.route("/generate", methods=["POST"])
def generate():
    purpose = request.form.get("purpose", "") # With '.get()' we are able to get the text inside 'purpose' in such a way that we are able to manipulate it as if the 'purpose' we can assign and empty string (""). 
    client_type = request.form.get("client_type", "")
    tone = request.form.get("tone", "")
    key_points = request.form.get("key_points", "")
    past_text = request.form.get("past_text", "")
    top_k = int(request.form.get("top_k", 2)) # If top_k is not provided in the 'POST', set the value of 2 as default.
    max_new_tokens = int(request.form.get("max_new_tokens", 220))
    max_new_tokens = min(max_new_tokens, 1024)
    
    # Preparing inputs
    key_points_list = [kp.strip() for kp in key_points.splitlines() if kp.strip()] # '.splitlines()' splits the key points separated by (\n) by the user into separate strings in the list. 'if kp.strip()' is used because: as you know '.strip()' removes the empty spaces from start and end of the string, now if a string is empty from inside or has empty spaces inside, then .strip() will make "  " into "". Now 'if' returns true only when string is non-empty, i.e., has some characters inside. Now, if the string is empty ("") after .strip(), that particular string won't be stored inside the key_points_list.
    key_points_text = "; ".join(key_points_list) # This converts all the key_points' strings into a single string separated by '; ' so that the model understand the key points in a contiguous manner.


    #  HANDLING FILE INGESTION: txt & pdf
    uploaded_files = request.files.getlist("documents")
    raw_docs=[]
    for f in uploaded_files:
        if not f or f.filename == "":
            continue
        filename = f.filename.lower()
        if filename.endswith(".txt"):
            try: 
                f.seek(0) # See the below 'f.seek(0)' to know about it.
                raw_docs.append(f.read().decode("utf-8"))
            except Exception:
                f.seek(0)
                raw_docs.append(f.read().decode("latin-1")) # fallback: read as latin-1. 'fallback' means backup.
        
        elif filename.endswith(".pdf"):
            try:
                import PyPDF2
                try:
                    f.seek(0) # The upload PDF file can have the pointer which is pointing to the page which the user was reading. To set the pointer of the PDF to the first page, we use f.seek(0) to reset the pdf file's pointer to 0 which is the start of that pdf file. If flask reject 'f.seek(0)' pass the file with the current pointer only.
                except Exception:
                    pass
                reader = PyPDF2.PdfReader(f)
                text = ""
                for i, page in enumerate(reader.pages): # Iterating through the pages in the PDF.
                    page_text = page.extract_text() or ""
                    text += f"[PAGE {i+1}]\n" + page_text + "\n\n" # Appends the page_text along with its page number. Giving page numbers is required so that correct context is capture when user also specify the page number / or there is a mention of page number in of the pdf's page in the past_emails / past_text.
                raw_docs.append(text)
            except Exception as e:
                print("PDF read error:", e) # If error occurs, print that error.


    if raw_docs:
        past_text = (past_text.strip() + "\n\n" + "\n\n".join(raw_docs)).strip()
    
    past_emails = [p.strip() for p in re.split(r'\n{2,}', past_text.strip()) if p.strip()]  # Past text js a single string that contains past emails. If there are blank lines (\n\n) in the end and start of the entire past email text, then past_text.strip() removes it. 'past_text.strip()' is the input string in 're.split()'. Usually, the user separates the emails with 2 or more blank lines. "r'\n{2,}'" splits the string into sub strings which are the separated emails stored as a string in the past_email list. 'if p.strip()' does the same thing as discussed above.
    query_for_retrieval = purpose + " " + key_points_text


    # Retrieve relevant context
    context_text = retrieve_context(bert, past_emails, query_for_retrieval, top_k=top_k) # Passing arguments to get the context_text.


    # Building prompt
    prompt = PROMPT_TEMPLATE.format(
        tone = tone,
        client_type = client_type,
        purpose = purpose,
        past = context_text or "No previous conversation provided.", # If empty use this message.
        key_points = key_points_text
    )


    # Generating outputs
    variants = generate_email_gpt(
        tokenizer, 
        gpt_model,
        prompt,
        max_new_tokens = max_new_tokens
    )

    return render_template("result.html", variants = variants, prompt=prompt) # Send the entire generated email along with the prompt into the results.html to display the results.


# ----------------------------------
# DOWNLOAD INDIVIDUAL EMAIL
# ----------------------------------
@app.route("/download", methods=["POST"]) # To enable download of email so that the individual can download the generated email.
def download():
    text = request.form.get("text", "") # Get the output email.
    tmp = tempfile.NamedTemporaryFile(delete = False, suffix=".txt") # A temporary file will be created in the browser that stores the generate email temporarily. 'delete=False' prevents auto deletion of the generated email. '.suffix=".txt"' is the file in which the email will be stored.
    tmp.write(text.encode()) # 'encode' writes the email into bytes in order to save it in that temporary file.
    tmp.close() # Closes the temp file.
    return send_file(tmp.name, as_attachment=True, download_name="email.txt") # Sends the email as an attachment into the user's browser with the default name 'email.txt'.


# Run Server
if __name__ == "__main__":
    print("Starting Flask......")
    app.run(debug=True, use_reloader=False) # 'use_reloader=False' prevents flask to load the model again.

import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import re
import faiss
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer
from difflib import SequenceMatcher

#  Local fallback for sentence splitting (no NLTK)
def split_into_sentences(text):
    sentence_endings = re.compile(r'(?<=[.!?])\s+')
    return sentence_endings.split(text.strip())

#  Load models from local paths (from preload_models.py)
summarizer_model = T5ForConditionalGeneration.from_pretrained("models/t5-base")
summarizer_tokenizer = T5Tokenizer.from_pretrained("models/t5-base")
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer)

qg_model = T5ForConditionalGeneration.from_pretrained("models/t5-qg")
qg_tokenizer = T5Tokenizer.from_pretrained("models/t5-qg")
qg_pipeline = pipeline("text2text-generation", model=qg_model, tokenizer=qg_tokenizer)

embedder = SentenceTransformer("models/minilm")

#  Streamlit config
st.set_page_config(page_title="EZ GenAI Smart Assistant", layout="wide")
st.title(" EZ Research Assistant")

# ** Memory Handling
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = []

# 📄 Extract text from PDF or TXT
def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")
    elif file.name.endswith(".pdf"):
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return " ".join(page.get_text() for page in doc)
    return ""

#** Generate summary
def summarize_text(text):
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    summary = summarizer(chunks[0], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    return " ".join(summary.split()[:150])  # Limit to ~150 words

# ** Ask Anything from doc
def ask_question(doc_text, question):
    sentences = split_into_sentences(doc_text)
    sentence_embeddings = embedder.encode(sentences).astype(np.float32)

    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(sentence_embeddings)

    q_vector = embedder.encode([question]).astype(np.float32)
    _, I = index.search(q_vector, k=1)
    ref = sentences[I[0][0]]

    answer = summarizer(ref, max_length=80, min_length=20, do_sample=False)[0]['summary_text']
    st.session_state.chat_memory.append({"q": question, "a": answer})

    return answer, ref

# ** Challenge Me logic
def generate_challenges(text, n=3):
    top_sentences = sorted(split_into_sentences(text), key=len, reverse=True)[:n]
    questions = []
    for sent in top_sentences:
        q = qg_pipeline(f"generate question: {sent}")[0]["generated_text"]
        questions.append((q, sent))
    return questions

# ✅ Fuzzy answer match
def evaluate_answer(user, reference):
    return SequenceMatcher(None, user.lower(), reference.lower()).ratio() > 0.6

# 🌐 Upload and interact
uploaded_file = st.file_uploader("📄 Upload a PDF or TXT file", type=["pdf", "txt"])

if uploaded_file:
    doc_text = extract_text(uploaded_file)

    st.subheader(" **Auto Summary**")
    st.info(summarize_text(doc_text))

    mode = st.radio("Select Mode", ["Ask Anything", "Challenge Me"])

    if mode == "Ask Anything":
        query = st.text_input(" Ask a question from the document:")
        if query:
            answer, reference = ask_question(doc_text, query)
            st.success(f"Answer: {answer}")
            st.caption(" Source Snippet (Answer Highlight):")
            st.code(reference)

        if st.session_state.chat_memory:
            with st.expander(" Chat Memory"):
                for turn in st.session_state.chat_memory[-5:]:
                    st.markdown(f"**You**: {turn['q']}")
                    st.markdown(f"**AI**: {turn['a']}")

    elif mode == "Challenge Me":
        st.subheader(" Logic Questions from the Document")
        challenges = generate_challenges(doc_text)

        for i, (q, ref) in enumerate(challenges):
            user_input = st.text_input(f"Q{i+1}: {q}", key=f"challenge_{i}")
            if user_input:
                correct = evaluate_answer(user_input, ref)
                st.markdown(f"**Your Answer:** {user_input}")
                if correct:
                    st.success(" Correct!")
                else:
                    st.error("Incorrect.")
                    st.caption("**Correct Answer Reference**:")
                    st.code(ref)

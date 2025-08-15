#   GenAI Smart Research Assistant

## Features

###  Document Understanding
- Upload a **PDF or TXT** file
- Automatically extracts and processes text

### Auto Summarization
- Generates a clean, ~150-word summary using `t5-base`

### Ask Anything
- Ask any question from the uploaded document
- The assistant retrieves a relevant passage using **FAISS similarity search**
- Summarizes the answer with context highlighting

### Challenge Me
- Generates 3 logic-based questions using a QG model (`valhalla/t5-qg`)
- User can attempt to answer
- System shows whether the answer is correct and also displays the correct snippet if wrong

### Bonus: Memory Handling
- Maintains context of last 5 Q&A turns in "Ask Anything" mode

##  Tech Stack

| Component        | Technology                                      |
|------------------|--------------------------------------------------|
| UI/Frontend      | Streamlit                                        |
| Summarization    | HuggingFace `t5-base`                            |
| Question Gen     | `valhalla/t5-base-qg-hl`                         |
| Semantic Search  | `sentence-transformers/all-MiniLM-L6-v2` + FAISS |
| PDF Handling     | `PyMuPDF`                                        |
| Fallback NLP     | Regex-based sentence splitting (no NLTK)         |



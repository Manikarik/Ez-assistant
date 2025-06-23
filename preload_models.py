# preload_models.py
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# Save summarizer
model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")
model.save_pretrained("models/t5-base")
tokenizer.save_pretrained("models/t5-base")

# Save QG model
qg_model = T5ForConditionalGeneration.from_pretrained("valhalla/t5-base-qg-hl")
qg_tokenizer = T5Tokenizer.from_pretrained("valhalla/t5-base-qg-hl")
qg_model.save_pretrained("models/t5-qg")
qg_tokenizer.save_pretrained("models/t5-qg")

# Save embedding model
SentenceTransformer("all-MiniLM-L6-v2").save("models/minilm")

print("✅ All models saved locally in 'models/' folder.")

print("🔄 Downloading summarizer (t5-base)...")
_ = pipeline("summarization", model="t5-base")

print("🔄 Downloading question generator (valhalla/t5-base-qg-hl)...")
_ = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

print("🔄 Downloading embedding model (MiniLM)...")
_ = SentenceTransformer("all-MiniLM-L6-v2")

print("✅ All models downloaded and cached locally.")

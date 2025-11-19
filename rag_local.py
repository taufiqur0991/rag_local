from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp

# PERBAIKAN: import yang benar & tidak deprecated
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate  

# 1. Load PDF
pdf_path = "./dokumens/gardening.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# 2. Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
texts = text_splitter.split_documents(documents)
print(f"Total chunk: {len(texts)}")

# 3. Embedding lokal (offline)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Vector DB Chroma
db_path = "./chroma_db"
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding_model,
    persist_directory=db_path
)
# vectordb.persist() → sudah tidak perlu lagi di versi baru Chroma
print("Vector database siap di folder:", db_path)

# 5. Load model GGUF
llm = LlamaCpp(
    model_path="./models/Phi-3-mini-4k-instruct-q4.gguf",  # PASTIKAN ADA .gguf
    temperature=0.7,
    max_tokens=1024,
    n_ctx=4096,
    n_gpu_layers=-1,   # -1 = gunakan semua layer di GPU (jika ada), kalau CPU pakai 0
    n_batch=512,
    verbose=True,      # ubah ke True dulu biar tahu proses loading
)

# 6. Prompt Template — INI YANG PALING PENTING! GANTI {crime} jadi {question}
template = """Gunakan HANYA informasi dari konteks berikut untuk menjawab pertanyaan.
Jika tidak tahu atau tidak ada di dokumen, katakan saja "Saya tidak tahu".

Konteks:
{context}

Pertanyaan: {question}

Jawaban langsung (dalam bahasa yang sama dengan pertanyaan):"""

# Tambahkan input_variables agar LangChain tahu variabel apa yang dipakai
QA_CHAIN_PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 7. Buat RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectordb.as_retriever(search_kwargs={"k": 4}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}   # pakai prompt yang sudah benar
)

# 8. Tanya jawab — pakai .invoke() (bukan __call__)
print("\n🤖 Chatbot Gardening RAG siap! 100% lokal & offline 🌱")
print("Ketik 'exit' atau 'keluar' untuk berhenti.\n")

while True:
    query = input("💬 Pertanyaan: ").strip()
    if query.lower() in ["exit", "quit", "keluar", ""]:
        print("Sampai jumpa! 👋")
        break

    # Ini cara yang benar di LangChain terbaru
    result = qa_chain.invoke({"query": query})

    print("\n🤖 Jawaban:")
    print(result["result"])

    print("\n📚 Sumber dari PDF:")
    for i, doc in enumerate(result["source_documents"], 1):
        page = doc.metadata.get("page", 0) + 1   # PyPDFLoader mulai dari 0
        snippet = doc.page_content.replace("\n", " ")[:200]
        print(f"   {i}. Halaman {page} → {snippet}...")
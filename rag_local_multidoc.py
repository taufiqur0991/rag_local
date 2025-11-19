import os
from glob import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings

# --- ⚙️ Konfigurasi ---
# Tentukan path model GGUF dan nama file
GGUF_MODEL_PATH = "./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" 
# Ganti dengan path model Anda

# Tentukan direktori tempat semua dokumen (PDF, DOCX) Anda berada
DOCUMENT_DIR = "./dokumens" 
# Ganti dengan path folder Anda

# Direktori untuk menyimpan Vector Database
CHROMA_DB_PATH = "./chroma_db_multidoc"
# --- ---------------- ---

def load_all_documents(directory_path):
    """Memuat semua dokumen PDF dan DOCX dari sebuah direktori."""
    all_docs = []
    
    # 1. Temukan semua file PDF dan DOCX
    pdf_files = glob(os.path.join(directory_path, "*.pdf"))
    docx_files = glob(os.path.join(directory_path, "*.docx"))
    
    all_files = pdf_files + docx_files
    
    if not all_files:
        print(f"‼️ Tidak ditemukan file .pdf atau .docx di: {directory_path}")
        return all_docs

    print(f"Ditemukan {len(all_files)} dokumen untuk diproses.")

    # 2. Muat setiap file
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        print(f" -> Memuat: {file_name}")
        
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
            
            # Memuat dan menambahkan ke daftar utama
            loaded_data = loader.load()
            
            # Tambahkan metadata nama file ke setiap chunk
            for doc in loaded_data:
                doc.metadata["source"] = file_name
            
            all_docs.extend(loaded_data)
            
        except Exception as e:
            print(f"   ⚠️ Gagal memuat file {file_name}. Error: {e}")
            continue

    return all_docs

def setup_vector_store(docs, embedding_model):
    """Memecah dokumen menjadi chunks dan menyimpannya ke ChromaDB."""
    if not docs:
        print("Tidak ada konten dokumen yang valid untuk diolah.")
        return None
        
    print("Memecah dokumen menjadi chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"Total chunks: {len(splits)}")
    print("Membuat dan menyimpan vector store (ChromaDB)...")
    
    # Membuat ChromaDB dari chunks dan embedding
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=CHROMA_DB_PATH
    )
    
    # Menyimpan database ke disk
    vectorstore.persist()
    print(f"Vector store berhasil disimpan di: {CHROMA_DB_PATH}")
    return vectorstore

def main():
    # 1. Inisialisasi Model Lokal (GGUF)
    try:
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            temperature=0.1,
            max_tokens=2048,
            n_gpu_layers=-1, # Atur > 0 jika Anda memiliki GPU yang kompatibel
            verbose=False,
            n_ctx=4096 
        )
        
        # --- 2. Inisialisasi Model Embedding yang konsisten (Misalnya, BGE Small EN) ---
        print("Memuat model Embedding (384-dim)...")
        # Ini akan mengunduh model kecil yang cepat dan konsisten (dimensi 384)
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    except Exception as e:
        print(f"‼️ Gagal memuat model GGUF. Pastikan path model benar.")
        print(f"Error: {e}")
        return

    # 2. Proses Dokumen (Memuat atau Membangun Database)
    if not os.path.exists(CHROMA_DB_PATH):
        try:
            print(f"Memuat dokumen dari direktori: {DOCUMENT_DIR}")
            docs = load_all_documents(DOCUMENT_DIR)
            if not docs:
                return
            vectorstore = setup_vector_store(docs, embedding_model)
        except Exception as e:
            print(f"‼️ Gagal memuat atau memproses dokumen. Pastikan folder dan isinya benar.")
            print(f"Error: {e}")
            return
    else:
        # Jika database sudah ada, muat saja
        print(f"Memuat vector store yang sudah ada dari: {CHROMA_DB_PATH}")
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding_model)

    # Pastikan vectorstore berhasil dimuat sebelum melanjutkan
    if not vectorstore:
        return

    # 3. Buat Chain RAG
    print("Menyiapkan RetrievalQA Chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}) # k=3 mengambil 3 chunk paling relevan
    )

    # 4. Loop Tanya Jawab
    print("\n--- Sistem RAG Lokal Multi-Dokumen Siap! ---")
    print(f"Database berisi data dari dokumen di folder: {DOCUMENT_DIR}")
    print("Ketik 'exit' atau 'quit' untuk keluar.")
    
    while True:
        query = input("❓ Pertanyaan Anda: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        if query:
            print("\n🔍 Mencari jawaban (Membutuhkan waktu, tergantung kecepatan model)...")
            try:
                # Memanggil Chain RAG
                result = qa_chain.invoke({"query": query})
                print(f"\n💡 Jawaban: {result['result']}")
                
                # OPTIONAL: Tunjukkan sumber dokumen (metadata)
                retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)
                sources = set(doc.metadata.get("source") for doc in retrieved_docs if "source" in doc.metadata)
                if sources:
                    print(f"\n📚 Sumber dokumen yang digunakan: {', '.join(sources)}")
                    
            except Exception as e:
                print(f"‼️ Terjadi kesalahan saat memanggil LLM: {e}")
        print("\n" + "="*70)

if __name__ == "__main__":
    main()
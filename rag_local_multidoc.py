import os
import time
from glob import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.llms import LlamaCpp
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate 

# --- ⚙️ Konfigurasi ---
# Tentukan path model GGUF dan nama file
GGUF_MODEL_PATH = "./models/Phi-3-mini-4k-instruct-q4.gguf" 
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
    #vectorstore.persist()
    print(f"Vector store berhasil disimpan di: {CHROMA_DB_PATH}")
    return vectorstore

def process_query(response):
    print('\n\n--- Sumber Dokumen ---')
    for doc in response['source_documents']:
        print(doc.metadata['source'])

def main():
    # 1. Inisialisasi Model Lokal (GGUF)
    try:
        llm = LlamaCpp(
            model_path=GGUF_MODEL_PATH,
            temperature=0.7,
            max_tokens=2048,
            n_gpu_layers=-1, # Atur > 0 jika Anda memiliki GPU yang kompatibel
            verbose=False,
            n_ctx=4096 
        )
        
        # --- 2. Inisialisasi Model Embedding yang konsisten (Misalnya, BGE Small EN) ---
        print("Memuat model Embedding (384-dim)...")
        # Ini akan mengunduh model kecil yang cepat dan konsisten (dimensi 384)
        embedding_model = HuggingFaceEmbeddings(
            #model_name="BAAI/bge-small-en-v1.5",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
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
    # Siapkan template prompt untuk RAG
    template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    Context: {context}
    Question: {question}
    """

    # Tambahkan input_variables agar LangChain tahu variabel apa yang dipakai
    QA_CHAIN_PROMPT = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    # 3. Buat Chain RAG
    print("Menyiapkan RetrievalQA Chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,        
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
    )

    # 4. Loop Tanya Jawab
    print("\n--- Sistem RAG Lokal Multi-Dokumen Siap! ---")
    print(f"Database berisi data dari dokumen di folder: {DOCUMENT_DIR}")
    print("Ketik 'exit' atau 'quit' untuk keluar.")
    
    while True:
        query = input("Pertanyaan Anda: ").strip()
        if query.lower() in ["exit", "quit","keluar"]:
            break
        print("\n 🔍 Mencari jawaban (Membutuhkan waktu, tergantung kecepatan model)...")
        start_q = time.time()
        # Memanggil Chain RAG
        result = qa_chain.invoke({"query": query})          
        print(result["result"])        
        print(f"\n ⏱️ Waktu menjawab: {time.time() - start_q:.2f} detik")                
        process_query(result)


if __name__ == "__main__":
    main()
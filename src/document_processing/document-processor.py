import os
import json
from typing import List, Dict, Any
import pypdf
import ebooklib
from ebooklib import epub
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF files."""
    with open(file_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def extract_text_from_epub(file_path: str) -> str:
    """Extract text from EPUB files."""
    book = epub.read_epub(file_path)
    text = ""
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            content = item.get_content().decode('utf-8')
            # Simple HTML tag removal (can be improved with BeautifulSoup)
            text += content.replace('<', ' <').replace('>', '> ').replace('\n', ' ') + "\n"
    return text

def load_web_content(urls: List[str]) -> List[str]:
    """Load content from websites."""
    texts = []
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            documents = loader.load()
            texts.extend([doc.page_content for doc in documents])
        except Exception as e:
            print(f"Error loading {url}: {e}")
    return texts

def load_wikipedia_content(topics: List[str]) -> List[str]:
    """Load content from Wikipedia."""
    texts = []
    for topic in topics:
        try:
            loader = WikipediaLoader(query=topic, lang="en")
            documents = loader.load()
            texts.extend([doc.page_content for doc in documents])
        except Exception as e:
            print(f"Error loading Wikipedia topic {topic}: {e}")
    return texts

def process_documents(config_path: str) -> None:
    """Process all documents and create vector database."""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize document storage
    all_texts = []
    
    # Process PDFs
    pdf_dir = config.get("pdf_directory", "data/pdfs")
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing PDF: {filename}")
            text = extract_text_from_pdf(file_path)
            all_texts.append({"content": text, "source": filename})
    
    # Process ebooks
    ebook_dir = config.get("ebook_directory", "data/ebooks")
    for filename in os.listdir(ebook_dir):
        if filename.endswith(('.epub', '.mobi', '.azw')):
            file_path = os.path.join(ebook_dir, filename)
            print(f"Processing ebook: {filename}")
            if filename.endswith('.epub'):
                text = extract_text_from_epub(file_path)
                all_texts.append({"content": text, "source": filename})
    
    # Process web content
    web_urls = config.get("web_urls", [])
    if web_urls:
        print(f"Processing {len(web_urls)} web pages")
        web_texts = load_web_content(web_urls)
        for i, text in enumerate(web_texts):
            all_texts.append({"content": text, "source": web_urls[i]})
    
    # Process Wikipedia topics
    wiki_topics = config.get("wikipedia_topics", [])
    if wiki_topics:
        print(f"Processing {len(wiki_topics)} Wikipedia topics")
        wiki_texts = load_wikipedia_content(wiki_topics)
        for i, text in enumerate(wiki_texts):
            all_texts.append({"content": text, "source": f"Wikipedia: {wiki_topics[i]}"})
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    documents = []
    for item in all_texts:
        chunks = text_splitter.split_text(item["content"])
        for chunk in chunks:
            documents.append({"content": chunk, "source": item["source"]})
    
    print(f"Created {len(documents)} document chunks")
    
    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    
    # Create vector store
    texts = [doc["content"] for doc in documents]
    metadatas = [{"source": doc["source"]} for doc in documents]
    
    db = Chroma.from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embedding_model,
        persist_directory=config.get("vector_db_path", "data/vectordb")
    )
    
    print(f"Vector database created and saved to {config.get('vector_db_path', 'data/vectordb')}")

if __name__ == "__main__":
    process_documents("config.json")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List, Dict, Any

class RAGService:
    """Retrieval Augmented Generation service for photo assessment."""
    
    def __init__(self, vector_db_path: str = "data/vectordb"):
        """Initialize RAG service with vector database path."""
        self.vector_db_path = vector_db_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.db = None
    
    def initialize(self):
        """Initialize the vector database connection."""
        if self.db is None:
            try:
                self.db = Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embedding_model
                )
                print(f"Vector database loaded from {self.vector_db_path}")
            except Exception as e:
                print(f"Error loading vector database: {e}")
                raise
    
    def get_relevant_content(self, query: str, image_analysis: Dict[str, Any], k: int = 5) -> List[Dict[str, str]]:
        """Retrieve relevant content from vector database based on query and image analysis."""
        self.initialize()
        
        # Create a more detailed query combining user question and image analysis
        enhanced_query = self._enhance_query(query, image_analysis)
        
        # Retrieve documents
        docs = self.db.similarity_search(enhanced_query, k=k)
        
        # Format results
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return results
    
    def _enhance_query(self, query: str, image_analysis: Dict[str, Any]) -> str:
        """Enhance the query with image analysis information."""
        # Extract key aspects from image analysis
        brightness = image_analysis.get("brightness", 0)
        contrast = image_analysis.get("contrast", 0)
        rule_of_thirds = image_analysis.get("rule_of_thirds", 0)
        sharpness = image_analysis.get("sharpness", 0)
        
        # Determine key aspects of the photo
        aspects = []
        
        if brightness < 80:
            aspects.append("dark")
        elif brightness > 180:
            aspects.append("bright")
            
        if contrast < 0.3:
            aspects.append("low contrast")
        elif contrast > 0.7:
            aspects.append("high contrast")
            
        if rule_of_thirds < 0.3:
            aspects.append("poor composition")
        elif rule_of_thirds > 0.6:
            aspects.append("good composition")
            
        if sharpness < 100:
            aspects.append("blurry")
        elif sharpness > 500:
            aspects.append("sharp")
            
        if image_analysis.get("faces", 0) > 0:
            aspects.append("portrait")
        
        # Create enhanced query
        enhanced_query = query
        if aspects:
            enhanced_query += f" The photo is {', '.join(aspects)}."
        
        return enhanced_query

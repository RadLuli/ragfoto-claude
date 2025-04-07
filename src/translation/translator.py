from transformers import MarianMTModel, MarianTokenizer
import torch

class Translator:
    """Translates text between English and Brazilian Portuguese."""
    
    def __init__(self, model_name="Helsinki-NLP/opus-mt-en-pt"):
        """Initialize translator with specific model."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.initialized = False
    
    def initialize(self):
        """Load model and tokenizer if not already loaded."""
        if not self.initialized:
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)
            self.model = MarianMTModel.from_pretrained(self.model_name)
            self.initialized = True
    
    def translate_to_portuguese(self, text):
        """Translate English text to Brazilian Portuguese."""
        self.initialize()
        
        # Handle empty text
        if not text:
            return ""
        
        # Split text into manageable chunks if too long
        max_length = 512
        chunks = self._split_text(text, max_length)
        translated_chunks = []
        
        for chunk in chunks:
            inputs = self.tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            
            # Generate translation
            with torch.no_grad():
                translated = self.model.generate(**inputs)
            
            # Decode the translation
            translated_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            translated_chunks.append(translated_text)
        
        # Join the chunks back together
        return " ".join(translated_chunks)
    
    def translate_to_english(self, text):
        """Translate Brazilian Portuguese text to English."""
        # For Portuguese to English, we need a different model
        pt_en_model_name = "Helsinki-NLP/opus-mt-pt-en"
        
        # Use a temporary tokenizer and model
        temp_tokenizer = MarianTokenizer.from_pretrained(pt_en_model_name)
        temp_model = MarianMTModel.from_pretrained(pt_en_model_name)
        
        # Handle empty text
        if not text:
            return ""
        
        # Split text into manageable chunks if too long
        max_length = 512
        chunks = self._split_text(text, max_length)
        translated_chunks = []
        
        for chunk in chunks:
            inputs = temp_tokenizer(chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            
            # Generate translation
            with torch.no_grad():
                translated = temp_model.generate(**inputs)
            
            # Decode the translation
            translated_text = temp_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            translated_chunks.append(translated_text)
        
        # Join the chunks back together
        return " ".join(translated_chunks)
    
    def _split_text(self, text, max_length):
        """Split text into chunks that won't exceed token limits."""
        words = text.split()
        chunks = []
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            # Roughly estimate if we're approaching the token limit
            if len(" ".join(current_chunk)) > max_length * 2:  # Conservative estimate
                chunks.append(" ".join(current_chunk))
                current_chunk = []
        
        # Add any remaining text
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

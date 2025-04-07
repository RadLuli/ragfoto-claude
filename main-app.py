import streamlit as st
import os
import tempfile
import time
from PIL import Image
import json
from src.document_processing.process_documents import process_documents
from src.image_analysis.image_analyzer import ImageAnalyzer
from src.enhancement.image_enhancer import ImageEnhancer
from src.translation.translator import Translator
from src.llm_service import LLMService
from src.rag_service import RAGService

# Load configuration
with open("config.json", "r") as f:
    config = json.load(f)

# Initialize services
@st.cache_resource
def load_services():
    # Process documents if vectordb doesn't exist
    vectordb_path = config.get("vector_db_path", "data/vectordb")
    if not os.path.exists(vectordb_path):
        with st.spinner("Initializing document database... This may take a few minutes."):
            process_documents("config.json")
    
    image_analyzer = ImageAnalyzer()
    image_enhancer = ImageEnhancer()
    translator = Translator(config["translation"]["model"])
    llm_service = LLMService("config.json")
    rag_service = RAGService(vectordb_path)
    
    return {
        "image_analyzer": image_analyzer,
        "image_enhancer": image_enhancer,
        "translator": translator,
        "llm_service": llm_service,
        "rag_service": rag_service
    }

services = load_services()

# Set page title and configuration
st.set_page_config(
    page_title="Análise Fotográfica - Sistema de Avaliação",
    page_icon="📸",
    layout="wide"
)

# App title
st.title("📸 Sistema de Avaliação Fotográfica")
st.subheader("Carregue uma foto para análise e avaliação")

# File upload widget
uploaded_file = st.file_uploader("Escolha uma imagem", type=["jpg", "jpeg", "png"])

# Query text input
query = st.text_area("Algum comentário ou pergunta específica sobre sua foto? (Opcional)", height=100)

# Process the uploaded image
if uploaded_file is not None:
    # Display the original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Imagem Original")
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_path = temp_file.name
        temp_file.write(uploaded_file.getvalue())
    
    try:
        # Step 1: Analyze the image
        status_text.text("Analisando a imagem...")
        progress_bar.progress(10)
        
        image_analysis = services["image_analyzer"].analyze_image(temp_path)
        
        # Step 2: Translate query to English if provided
        status_text.text("Processando consulta...")
        progress_bar.progress(20)
        
        english_query = query
        if query:
            english_query = services["translator"].translate_to_english(query)
        
        # Step 3: Get relevant content from RAG system
        status_text.text("Consultando base de conhecimento...")
        progress_bar.progress(30)
        
        relevant_content = services["rag_service"].get_relevant_content(
            english_query or "Evaluate this photo",
            image_analysis
        )
        
        # Step 4: Generate assessment using LLM
        status_text.text("Gerando avaliação...")
        progress_bar.progress(50)
        
        assessment = services["llm_service"].generate_assessment(
            image_analysis,
            english_query or "Evaluate this photo",
            relevant_content
        )
        
        # Step 5: Generate enhancement suggestions
        status_text.text("Gerando sugestões de melhoria...")
        progress_bar.progress(60)
        
        suggestions = services["llm_service"].generate_suggestions(
            assessment,
            image_analysis
        )
        
        # Step 6: Translate assessment and suggestions to Portuguese
        status_text.text("Traduzindo avaliação para Português...")
        progress_bar.progress(70)
        
        portuguese_assessment = services["translator"].translate_to_portuguese(
            assessment["overall_assessment"]
        )
        
        portuguese_suggestions = []
        for suggestion in suggestions:
            portuguese_suggestions.append(
                services["translator"].translate_to_portuguese(suggestion)
            )
        
        # Step 7: Create enhanced image
        status_text.text("Gerando imagem aprimorada...")
        progress_bar.progress(80)
        
        enhanced_image_path = os.path.join(tempfile.gettempdir(), "enhanced_image.jpg")
        services["image_enhancer"].enhance_image(
            temp_path,
            image_analysis,
            suggestions,
            enhanced_image_path
        )
        
        # Create visualization of analysis
        analysis_viz_path = os.path.join(tempfile.gettempdir(), "analysis_viz.jpg")
        services["image_analyzer"].save_analysis_visualization(
            temp_path,
            analysis_viz_path
        )
        
        # Step 8: Display results
        status_text.text("Concluído!")
        progress_bar.progress(100)
        
        # Display the assessment
        st.subheader("📝 Avaliação")
        st.write(portuguese_assessment)
        
        # Display the score with stars
        st.subheader("⭐ Pontuação")
        stars = int(assessment["score"])
        half_star = (assessment["score"] - stars) >= 0.5
        
        star_html = "".join(["⭐" for _ in range(stars)])
        if half_star:
            star_html += "✨"
        
        st.markdown(f"**{assessment['score']}/5** {star_html}")
        
        # Display individual criteria scores if available
        if "criteria_scores" in assessment and assessment["criteria_scores"]:
            st.subheader("Pontuação por Critério")
            for criterion, score in assessment["criteria_scores"].items():
                st.write(f"**{criterion.replace('_', ' ').title()}:** {score}/5")
        
        # Display enhancement suggestions
        st.subheader("💡 Sugestões de Melhoria")
        for suggestion in portuguese_suggestions:
            st.markdown(f"- {suggestion}")
        
        # Display the enhanced image
        with col2:
            st.subheader("Imagem Aprimorada")
            enhanced_image = Image.open(enhanced_image_path)
            st.image(enhanced_image, use_column_width=True)
        
        # Display the analysis visualization
        st.subheader("Visualização da Análise")
        analysis_viz = Image.open(analysis_viz_path)
        st.image(analysis_viz, use_column_width=False)
        
    except Exception as e:
        st.error(f"Ocorreu um erro ao processar a imagem: {str(e)}")
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# Information about the system
with st.expander("ℹ️ Sobre este Sistema"):
    st.write("""
    Este sistema utiliza técnicas avançadas de processamento de imagem e inteligência artificial 
    para avaliar fotografias com base em princípios fotográficos profissionais.
    
    O sistema analisa aspectos como:
    - Composição (incluindo regra dos terços)
    - Iluminação e exposição
    - Contraste e equilíbrio de cores
    - Nitidez e foco
    - Outros elementos técnicos e artísticos
    
    As sugestões de melhoria são geradas com base em bibliografia especializada e conhecimento fotográfico.
    """)

# Footer
st.markdown("---")
st.markdown("📸 Desenvolvido para suporte educacional em fotografia.")

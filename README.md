# Sistema de Avaliação Fotográfica - RAG Multimodal

Sistema de avaliação fotográfica baseado em RAG (Retrieval Augmented Generation) para fornecer feedback educacional em tempo real para estudantes de fotografia.

## Características

- Análise técnica de imagens fotográficas
- Avaliação baseada em referências bibliográficas e educacionais
- Feedback em Português Brasileiro
- Sugestões específicas para melhoria
- Geração de imagens aprimoradas
- Interface web de fácil utilização

## Requisitos

- Python 3.8+
- Mac ou Linux (preferível) / Windows
- Conexão à internet para instalação inicial
- Aproximadamente 2GB de espaço em disco

## Configuração

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/avaliacao-fotografica.git
   cd avaliacao-fotografica
   ```

2. Crie e ative um ambiente virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # No Mac/Linux
   # ou
   .\venv\Scripts\activate  # No Windows
   ```

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

4. Instale o Ollama:
   - Visite [ollama.ai](https://ollama.ai/) e baixe o instalador para seu sistema
   - Siga as instruções de instalação

5. Baixe o modelo LLaMA:
   ```bash
   ollama pull llama3
   ```

6. Organize seus documentos de referência:
   - Coloque arquivos PDF em `data/pdfs/`
   - Coloque e-books (EPUB, MOBI, AZW) em `data/ebooks/`
   - Edite `config.json` para adicionar URLs e tópicos da Wikipedia

## Uso Local

1. Inicie a aplicação:
   ```bash
   streamlit run app.py
   ```

2. Acesse no navegador:
   - A interface estará disponível em `http://localhost:8501`

## Implantação

Para implantar em um servidor online gratuito:

1. Crie uma conta no Streamlit Cloud:
   - Visite [streamlit.io](https://streamlit.io/) e registre-se

2. Conecte sua conta do GitHub:
   - Faça upload do projeto para um repositório GitHub
   - Conecte o repositório ao Streamlit Cloud

3. Configure a implantação:
   - Selecione o arquivo principal como `app.py`
   - Defina os requisitos de pacotes (requirements.txt)
   - Inicie a implantação

Para servidores alternativos:
- O sistema pode ser hospedado em serviços como Heroku, Render ou PythonAnywhere
- Consulte a documentação específica de cada serviço para detalhes

## Personalização

- Edite `config.json` para:
  - Modificar critérios de avaliação
  - Ajustar parâmetros do modelo de linguagem
  - Configurar parâmetros de melhoria de imagem
  - Adicionar mais fontes de referência

## Limitações

- Processamento inicial pode ser lento em máquinas com recursos limitados
- Para grandes volumes de imagens, considere um serviço hospedado mais robusto
- Modelos de linguagem locais têm qualidade inferior a modelos comerciais

## Licença

Este projeto é disponibilizado sob a licença MIT. Veja o arquivo LICENSE para detalhes.

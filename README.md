# Asistente Documento

## Preparar entorno
```bash
pip install -r requirements.txt
```

Añade a `.env` tus claves Azure OpenAI:
```
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_EMBEDDINGS_API_KEY=...
AZURE_EMBEDDINGS_ENDPOINT=...
```

## Crear índice
```bash
python create_doc_embeddings.py "procedimiento valvulas inserto sold.docx"  # crea directorio doc_faiss/
```

## Lanzar Streamlit
```bash
streamlit run doc_app.py
``` 
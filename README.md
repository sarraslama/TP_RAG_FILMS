# TP RAG - Recommandation de Films

## Description
Système RAG de recommandation de films basé sur le dataset TMDB 5000.

## Installation
\\ash
python -m venv venv
venv\Scriptsctivate
pip install -r requirements.txt
\
## Configuration
Créer un fichier .env :
\GROQ_API_KEY=votre_clé_ici
\
## Utilisation
1. Indexation (une seule fois) :
\\ash
python indexation.py
\
2. Lancer le RAG :
\\ash
python rag.py
\
## Outils utilisés
- Groq (LLM)
- FAISS (base vectorielle)
- sentence-transformers (embeddings)
- pandas (traitement des données)

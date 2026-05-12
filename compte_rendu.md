# Compte-rendu TP RAG Films

## Choix techniques

### Modele embedding
Jai choisi all-mpnet-base-v2 car le dataset TMDB est en anglais.
Ce modele produit des vecteurs de dimension 768.

### Chunking
Taille de 500 caracteres avec overlap de 50.
Chaque film est converti en texte structuré avec titre, annee, genres, note et synopsis.

### FAISS
Index sauvegarde sur disque pour eviter de reindexer a chaque lancement.

### LLM
llama-3.3-70b-versatile via Groq car llama3-8b-8192 est decommissionne.

## Difficultes rencontrees

### 1. Modele Groq decommissionne
llama3-8b-8192 nest plus disponible, remplace par llama-3.3-70b-versatile.

### 2. Fichier .env
PowerShell cree les fichiers en UTF-16, causant une erreur dotenv.
Resolu en creant le .env avec Python directement.

### 3. Donnees tabulaires
Le CSV contient des colonnes JSON imbriquees pour les genres.
Resolu avec json.loads() pour extraire les noms de genres.

## Resultats
- 4803 films charges
- 6968 chunks indexes dans FAISS
- Temps dindexation : 3 minutes environ
- Systeme repond en francais avec titres, notes et annees

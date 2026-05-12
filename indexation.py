import pandas as pd
import numpy as np
import faiss
import json
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ─────────────────────────────────────────

def extraire_genres(genres_json):
    """Extrait les noms de genres depuis le format JSON de TMDB."""
    try:
        genres = json.loads(genres_json)
        return ", ".join([g["name"] for g in genres])
    except:
        return ""

def preparer_documents(chemin_csv):
    """Charge le CSV et convertit chaque film en texte embedable."""
    df = pd.read_csv(chemin_csv)
    print(f"✅ {len(df)} films chargés")

    # Garder uniquement les films avec un synopsis
    df = df[df["overview"].notna() & (df["overview"] != "")]
    print(f"✅ {len(df)} films avec synopsis")

    documents = []
    for _, row in df.iterrows():
        genres = extraire_genres(row.get("genres", "[]"))
        titre = str(row.get("title", ""))
        synopsis = str(row.get("overview", ""))
        note = row.get("vote_average", 0)
        annee = str(row.get("release_date", ""))[:4]
        langue = str(row.get("original_language", ""))
        duree = row.get("runtime", "")

        # Texte qui sera embedé
        texte = f"""Titre : {titre}
Année : {annee}
Genres : {genres}
Note : {note}/10
Langue originale : {langue}
Durée : {duree} minutes
Synopsis : {synopsis}"""

        documents.append({
            "id": str(row.get("id", "")),
            "contenu": texte,
            "metadata": {
                "titre": titre,
                "annee": annee,
                "note": float(note) if note else 0.0,
                "genres": genres,
                "langue": langue,
            }
        })

    return documents


# ─────────────────────────────────────────
# 2. CHUNKING
# ─────────────────────────────────────────

def chunker(texte, taille_max=500, overlap=50):
    """Découpe un texte en chunks avec chevauchement."""
    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille_max
        chunk = texte[debut:fin]
        chunks.append(chunk)
        debut += taille_max - overlap
    return chunks


def chunker_documents(documents):
    """Applique le chunking à tous les documents."""
    chunks_avec_meta = []
    for doc in documents:
        chunks = chunker(doc["contenu"], taille_max=500, overlap=50)
        for i, chunk in enumerate(chunks):
            chunks_avec_meta.append({
                "contenu": chunk,
                "metadata": doc["metadata"],
                "chunk_id": f"{doc['id']}_chunk_{i}"
            })
    print(f"✅ {len(chunks_avec_meta)} chunks créés")
    return chunks_avec_meta


# ─────────────────────────────────────────
# 3. EMBEDDINGS
# ─────────────────────────────────────────

def embedder_chunks(chunks_avec_meta, modele):
    """Transforme les chunks en vecteurs."""
    textes = [c["contenu"] for c in chunks_avec_meta]
    print("⏳ Création des embeddings (peut prendre quelques minutes)...")
    vecteurs = modele.encode(textes, show_progress_bar=True)
    print(f"✅ Embeddings créés — dimension : {vecteurs.shape}")
    return np.array(vecteurs, dtype=np.float32)


# ─────────────────────────────────────────
# 4. INDEX FAISS
# ─────────────────────────────────────────

def creer_index_faiss(vecteurs):
    """Crée un index FAISS à partir des vecteurs."""
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    print(f"✅ Index FAISS créé — {index.ntotal} vecteurs indexés")
    return index

def sauvegarder_index(index, chunks_avec_meta, chemin="index_films"):
    """Sauvegarde l'index FAISS et les métadonnées."""
    os.makedirs(chemin, exist_ok=True)
    faiss.write_index(index, f"{chemin}/index.faiss")
    with open(f"{chemin}/chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_avec_meta, f, ensure_ascii=False, indent=2)
    print(f"✅ Index sauvegardé dans '{chemin}/'")


# ─────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Chargement
    documents = preparer_documents("data/tmdb_5000_movies.csv")

    # Chunking
    chunks_avec_meta = chunker_documents(documents)

    # Modèle d'embedding (multilingue)
    modele = SentenceTransformer("all-mpnet-base-v2")

    # Embeddings
    vecteurs = embedder_chunks(chunks_avec_meta, modele)

    # Index FAISS
    index = creer_index_faiss(vecteurs)

    # Sauvegarde
    sauvegarder_index(index, chunks_avec_meta)

    print("\n🎬 Indexation terminée ! Tu peux maintenant lancer rag.py")
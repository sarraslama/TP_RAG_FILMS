import faiss
import json
import numpy as np
import os
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# 1. CHARGEMENT DE L'INDEX
# ─────────────────────────────────────────

def charger_index(chemin="index_films"):
    """Charge l'index FAISS et les métadonnées depuis le disque."""
    index = faiss.read_index(f"{chemin}/index.faiss")
    with open(f"{chemin}/chunks.json", "r", encoding="utf-8") as f:
        chunks_avec_meta = json.load(f)
    print(f"✅ Index chargé — {index.ntotal} vecteurs")
    print(f"✅ {len(chunks_avec_meta)} chunks chargés")
    return index, chunks_avec_meta


# ─────────────────────────────────────────
# 2. RECHERCHE VECTORIELLE
# ─────────────────────────────────────────

def rechercher(question, modele, index, chunks_avec_meta, k=5):
    """Recherche les k chunks les plus pertinents pour une question."""
    # Embedder la question
    vecteur_question = modele.encode([question])
    vecteur_question = np.array(vecteur_question, dtype=np.float32)

    # Recherche dans FAISS
    distances, indices = index.search(vecteur_question, k)

    resultats = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            chunk = chunks_avec_meta[idx]
            resultats.append({
                "contenu": chunk["contenu"],
                "metadata": chunk["metadata"],
                "score": float(distances[0][i])
            })
    return resultats


# ─────────────────────────────────────────
# 3. PROMPT SYSTÈME
# ─────────────────────────────────────────

def construire_prompt_systeme():
    """Retourne le prompt système pour l'assistant films."""
    return """Tu es un expert en cinéma passionné et un assistant de recommandation de films.

Ton rôle :
- Recommander des films pertinents basés UNIQUEMENT sur les informations fournies dans le contexte
- Pour chaque recommandation, toujours citer : le titre du film, l'année, la note sur 10, et les genres
- Expliquer pourquoi ce film correspond à la demande de l'utilisateur
- Si l'utilisateur précise une langue (français, anglais...), filtrer en conséquence

Règles importantes :
- Ne jamais inventer un film qui n'est pas dans le contexte fourni
- Si aucun film du contexte ne correspond, dire honnêtement "Je ne trouve pas de film correspondant dans ma base"
- Si l'utilisateur demande un film très récent (après 2017), préciser que ta base de données s'arrête en 2017
- Toujours répondre en français, même si les données sont en anglais
- Présenter les recommandations de manière claire et enthousiaste"""


# ─────────────────────────────────────────
# 4. GÉNÉRATION DE LA RÉPONSE
# ─────────────────────────────────────────

def generer_reponse(question, chunks_pertinents, client):
    """Génère une réponse avec Groq en utilisant les chunks comme contexte."""

    # Construire le contexte
    contexte = ""
    for i, chunk in enumerate(chunks_pertinents):
        meta = chunk["metadata"]
        contexte += f"""
--- Film {i+1} ---
{chunk['contenu']}
"""

    # Construire le prompt utilisateur
    prompt_utilisateur = f"""Voici des informations sur des films de ma base de données :

{contexte}

Question de l'utilisateur : {question}

Réponds en te basant uniquement sur les films fournis ci-dessus."""

    # Appel à l'API Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": construire_prompt_systeme()},
            {"role": "user", "content": prompt_utilisateur}
        ],
        temperature=0.7,
        max_tokens=1000
    )

    return response.choices[0].message.content


# ─────────────────────────────────────────
# 5. INTERFACE EN LIGNE DE COMMANDE
# ─────────────────────────────────────────

def main():
    print("🎬 Chargement du système RAG Films...")

    # Charger l'index
    index, chunks_avec_meta = charger_index()

    # Charger le modèle d'embedding
    print("⏳ Chargement du modèle d'embedding...")
    modele = SentenceTransformer("all-mpnet-base-v2")
    print("✅ Modèle chargé")

    # Initialiser le client Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    print("\n🎬 Système RAG prêt ! Tapez 'quit' pour quitter.\n")
    print("=" * 50)

    while True:
        question = input("\n🎬 Votre question : ").strip()

        if question.lower() in ["quit", "exit", "q"]:
            print("Au revoir ! 🎬")
            break

        if not question:
            continue

        print("\n⏳ Recherche en cours...")

        # Recherche des chunks pertinents
        chunks_pertinents = rechercher(question, modele, index, chunks_avec_meta, k=5)

        # Génération de la réponse
        reponse = generer_reponse(question, chunks_pertinents, client)

        # Affichage
        print("\n" + "=" * 50)
        print("🎬 RÉPONSE :")
        print("=" * 50)
        print(reponse)
        print("=" * 50)

        # Afficher les sources
        print("\n📽️ Sources utilisées :")
        titres_vus = []
        for chunk in chunks_pertinents:
            titre = chunk["metadata"]["titre"]
            note = chunk["metadata"]["note"]
            annee = chunk["metadata"]["annee"]
            if titre not in titres_vus:
                print(f"  - {titre} ({annee}) — Note : {note}/10")
                titres_vus.append(titre)


if __name__ == "__main__":
    main()
    
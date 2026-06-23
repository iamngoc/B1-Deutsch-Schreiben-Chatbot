"""
setup.py
Einmaliges Skript: verarbeitet alle PDFs und erstellt eine persistente
Chroma-Datenbank, die später von der Streamlit-App genutzt wird.

Ausführen mit: python setup_database.py
"""

import os
import json
import tempfile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# --- Konfiguration ---
JSON_FOLDER = "./jsons"            # Ordner mit allen PDF-Dateien
PERSIST_DIR = "./chroma_db"      # Ziel-Ordner für die Datenbank

def load_json_as_documents(filepath):
    """Lädt eine JSON-Datei mit B1-Schreibaufgaben und wandelt jeden Eintrag
    in ein LangChain-Document um."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for item in data:
        leitpunkte = item.get("leitpunkte", [])
        leitpunkte_text = "\n".join(f"- {lp}" for lp in leitpunkte)

        text = f"""Thema: {item.get('titel', '')}

Situation: {item.get('situation', '')}

Leitpunkte:
{leitpunkte_text}

Modellantwort:
{item.get('modellantwort', '')}"""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(filepath),
                    "thema_nr": item.get("thema_nr"),
                    "titel": item.get("titel"),
                }
            )
        )

    return documents

def main():
    # 1. Alle PDFs aus dem Ordner einlesen
    json_files = [f for f in os.listdir(JSON_FOLDER) if f.lower().endswith(".json")]

    if not json_files:
        print(f"Keine json-Dateien in '{JSON_FOLDER}' gefunden.")
        return

    print(f"Gefundene jsons: {json_files}")

    documents = []
    for json in json_files:
        filepath = os.path.join(JSON_FOLDER, json)
        loaded = load_json_as_documents(filepath)
        documents.extend(loaded)
        print(f"  -> {documents}: {len(loaded)} Themen geladen")
    print(f"Gesamt: {len(documents)} Themen/Dokumente")
    docs = documents
    # # 2. In Chunks aufteilen
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=1500,
    #     chunk_overlap=200
    # )
    #docs = text_splitter.split_documents(documents)

    # 3. Embeddings + Vektordatenbank erstellen und persistieren
    print("Erstelle Embeddings (kann einige Minuten dauern)...")
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')

    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )

    print(f"✅ Fertig! Datenbank gespeichert in '{PERSIST_DIR}'")
    print(f"   Anzahl Chunks: {len(docs)}")


if __name__ == "__main__":
    main()
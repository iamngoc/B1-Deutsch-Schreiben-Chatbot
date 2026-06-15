import shutil
import warnings



warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message="Special tokens have been added")

import torch
import streamlit as st
import tempfile
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import ConversationalRetrievalChain

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

st.set_page_config(page_title="B1-Schreiben Deutsch Assistant", layout="wide")
PERSIST_DIR = "./chroma_db"
st.title("📝 B1-Schreiben Deutsch Assistent")
st.markdown("""
<h2>Diese App hilft dir bei der Vorbereitung auf den B1-Schreibtest in Deutsch. Gib bitte ein Thema ein, über das du schreiben möchtest.</h2>

Der B1-Schreibtest besteht aus drei Unterabschnitten:

- Abschnitt 1: Schreiben Sie eine E-Mail (ca. 80 Wörter) im informellen Stil.
- Abschnitt 2: Äußern Sie Ihre persönliche Meinung (ca. 80 Wörter) zu einem Thema, nachdem Sie einen Kommentar oder eine Meinung dazu gelesen haben.
- Abschnitt 3: Schreiben Sie eine E-Mail (ca. 40 Wörter) im formellen oder halbformellen Stil.
""")

# Session state initialization
if "rag_chain" not in st.session_state:
    with st.spinner("Lade Modelle und Datenbank... "):
        embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-small')
        vector_db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings)
        retriever = vector_db.as_retriever(search_kwargs={"k": 6})
        model_name = 'Qwen/Qwen2.5-3B-Instruct'
        model = AutoModelForCausalLM.from_pretrained(model_name,
                                                     torch_dtype=torch.float32,
                                                     low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_pipeline = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=250,
            pad_token_id=tokenizer.eos_token_id
        )

        st.session_state.llm = HuggingFacePipeline(pipeline=model_pipeline)
        prompt = ChatPromptTemplate.from_template("""Du bist ein Sprachlehrer für Deutsch (B1) und hilfst auch auf Vietnamesisch (Bạn là giáo viên tiếng Đức, cũng có thể hỗ trợ bằng tiếng Việt).

            Beispieltexte / Tài liệu tham khảo:
            {context}

            Frage / Câu hỏi:
            {question}

            Schreibe NUR den Brief/die E-Mail im B1-Stil. Verwende eine passende Anrede und einen passenden Schluss.
            Verwende eine passende Anrede und einen passenden Schluss.
            Verwende keinen Doppelpunkt, keinen Gedankenstrich. 
            Stelle die Richtigkeit des Inhalts sicher sodass der Text sofort abgeschrieben werden ohne Korrigieren kann. 
            Schreibe KEINE Erklärungen, KEINE Bewertung, KEINE zusätzlichen Kommentare nach dem Text.
            Höre auf, sobald der Brief mit der Grußformel und dem Namen endet.
            
            Antwort:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        st.session_state.rag_chain = ({"context": retriever | format_docs,
                  "question": RunnablePassthrough()} | prompt | st.session_state.llm | StrOutputParser())

    st.success("Du kannst direkt Fragen stellen.")

def clean_answer(output: str) -> str:
    #Alles vor "Antwort:" entfernen
    parts = output.split("Antwort:")
    text = parts[1].strip() if len(parts) > 1 else output.strip()

    # Bekannte "Ramble"-Marker als harte Stopps definieren
    stop_markers = [
        "\nAssistant:",
        "\nDiese Antwort",
        "\nBitte beachten",
        "\nDer Text ist",
        "\nIch freue mich",
        "\nGuten Tag, Ihr Sprachlehrer",
        "\nFalls Sie weitere",
        "\nIch wünsche Ihnen",
        "\nDanke für Ihre",
        "\nIch bin hier",
    ]
    for marker in stop_markers:
        if marker in text:
            text = text.split(marker)[0]

    return text.strip()
# Fragebereich

user_question = st.text_area("Deine Aufgabe / Câu hỏi của bạn:")

if st.button("Antwort generieren") and user_question.strip():
    with st.spinner("Generiere Antwort..."):
        output = st.session_state.rag_chain.invoke(user_question)
        answer = clean_answer(output)
    st.markdown("### Antwort")
    st.write(answer)


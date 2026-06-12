import sqlite3
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

conn = sqlite3.connect("expanded_hospital.db")
cur = conn.cursor()

cur.execute("SELECT patient_id, visit_date, note FROM clinical_notes")
rows = cur.fetchall()

docs = [
    Document(
        page_content=note,
        metadata={"patient_id": pid, "date": date}
    )
    for pid, date, note in rows
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(docs, embeddings)
db.save_local("rag_index")

print("done")
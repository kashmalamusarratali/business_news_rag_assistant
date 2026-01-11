# =========================
# 1. IMPORTS
# =========================
import os
import requests
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_classic.chains import RetrievalQA


from langchain_community.vectorstores import Chroma

from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain



from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)


# =========================
# 2. ENV SETUP
# =========================
load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("Gem_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# =========================
# 3. RSS FEED INGESTION
# =========================
RSS_URL = "https://www.gulf-times.com/rssFeed/2"
feed = feedparser.parse(RSS_URL)

news_df = pd.DataFrame(columns=["title", "link", "published", "summary"])

for i, entry in enumerate(feed.entries):
    news_df.loc[i] = [
        entry.get("title", ""),
        entry.get("link", ""),
        entry.get("published", ""),
        entry.get("summary", "")
    ]


# =========================
# 4. ARTICLE SCRAPER
# =========================
def get_article_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        text = " ".join(p.get_text() for p in paragraphs)
        return text.strip()
    except Exception:
        return ""


news_df["article_text"] = news_df["link"].apply(get_article_text)
news_df = news_df[news_df["article_text"].str.len() > 200]

print(f"Fetched {len(news_df)} full articles")


# =========================
# 5. TEXT CHUNKING
# =========================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

documents = []
metadatas = []

for _, row in news_df.iterrows():
    chunks = text_splitter.split_text(row["article_text"])
    for chunk in chunks:
        documents.append(chunk)
        metadatas.append({
            "title": row["title"],
            "source": row["link"],
            "published": row["published"]
        })


# =========================
# 6. GEMINI EMBEDDINGS + VECTOR STORE
# =========================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

vector_db = Chroma.from_texts(
    texts=documents,
    embedding=embeddings,
    metadatas=metadatas,
    persist_directory="./financial_news_gemini_db"
)

vector_db.persist()

print("Vector database created successfully (Gemini embeddings)")


# =========================
# 7. GEMINI LLM + RAG CHAIN
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)


# =========================
# 8. CHAT LOOP
# =========================
print(" Financial News RAG Bot (Gemini) Ready")
print("Type 'exit' to quit\n")

while True:
    query = input("Ask a question: ")

    if query.lower() in ["exit", "quit"]:
        break

    result = qa_chain.invoke({"query": query})

    print("\nAnswer:")
    print(result["result"])

    print("\nSources:")
    for doc in result["source_documents"]:
        print("-", doc.metadata["title"])

    print("\n" + "-" * 50)

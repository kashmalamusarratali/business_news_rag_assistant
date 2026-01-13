# Business News RAG Assistant (Gemini + LangChain)

A Retrieval-Augmented Generation (RAG) system that ingests real-time business news from RSS feeds, semantically indexes full articles using Google Gemini embeddings, and enables grounded question-answering via a Gemini LLM using LangChain.

This project demonstrates end-to-end LLM application engineering, including data ingestion, text processing, vector databases, and LLM orchestration.

# Project Overview

Modern LLMs are powerful but suffer from hallucinations when asked about external or up-to-date information.
This project solves that by combining:

Real-time news ingestion

Semantic vector search

LLM-based answer generation grounded in retrieved context

High-Level Flow
RSS Feed → Article Scraping → Text Chunking
        → Gemini Embeddings → Vector Database (Chroma)
        → Retriever → Gemini LLM → Answer with Sources

#  Key Features

Live RSS ingestion (Gulf Times – business News)

Full article scraping using BeautifulSoup

Smart text chunking for optimal retrieval

Gemini embeddings for semantic understanding

Gemini LLM for accurate, grounded responses

Vector database (Chroma) for fast similarity search

Source attribution for transparency

Clean, modular, production-ready structure

# Tech Stack
Component	Technology
Language	Python
LLM	Google Gemini 2.5
Embeddings	Gemini embedding-004
Framework	LangChain
Vector DB	Chroma
Scraping	BeautifulSoup
Data Handling	Pandas
Config	python-dotenv

# Why RAG (Retrieval-Augmented Generation)?

Traditional LLM usage:

Hallucinates facts

No access to real-time data

No source traceability

RAG fixes this by:

Retrieving relevant documents

Feeding them as context to the LLM

Generating grounded, explainable answers

# Implementation Details
 **RSS Ingestion**

Uses feedparser to fetch business news feeds

Extracts title, link, publish date, and summary

 **Article Scraping**

Fetches full article content from each link

Parses <p> tags using BeautifulSoup

Filters out short or empty articles

**Text Chunking**

Uses RecursiveCharacterTextSplitter

Chunk size: 500 tokens

Overlap: 100 tokens

This improves retrieval accuracy and prevents context overflow.

**Embeddings (Gemini)**

Model: models/embedding-004

Converts text chunks into high-dimensional semantic vectors

Stored persistently in Chroma

**Vector Database**

Chroma used for local, fast similarity search

Metadata stored alongside each chunk:

Title

Source URL

Published date

**Retrieval + LLM**

Top-K relevant chunks retrieved per query

Passed to Gemini LLM (gemini-2.5-flash)

# Future Enhancements

Scheduled RSS ingestion (cron / Airflow)

FastAPI backend for web or chatbot UI

Conversational memory

PostgreSQL + pgvector

Agent-based tools (alerts, summaries, analytics)

# Use Cases

business research assistant

News intelligence platform

Market trend analysis

Policy & economic monitoring



# Author

Kashmala Musarrat Ali

LLM & Agentic AI Engineer

Specialized in RAG systems, Generative AI, Data Engineering & Analytics



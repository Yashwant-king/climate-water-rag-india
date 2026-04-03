# ClimateWater RAG (India) | AI Assistant

## Overview
This project is an advanced **Retrieval-Augmented Generation (RAG)** system and a custom evaluation framework designed for the **AI Engineer Intern Assignment 3**. It focuses on the highly specialized and niche domain of **"Impact of Climate Change on Indian Water Resources."**

The objective is to provide expert-level insights on Himalayan glacier melt, monsoon variability, and groundwater depletion while quantitatively and qualitatively assessing the accuracy of the AI-generated answers.

## Dataset Design Decision
For this project, multiple small documents were used instead of a single large file.

Although it is technically possible to store all information in one `.txt` file, this approach was avoided because it weakens the effectiveness of a Retrieval-Augmented Generation (RAG) system.

RAG systems rely on retrieving the most relevant pieces of information from a collection of documents. If all content is stored in a single file, the retrieval step becomes less meaningful, as the system always searches within the same source, reducing the ability to evaluate retrieval quality.

To ensure proper functioning and evaluation of the RAG pipeline, the dataset was structured as multiple documents (6–8 files), each focusing on a specific subtopic within the domain "Impact of climate change on water resources in India."

This design improves:
* Retrieval accuracy
* Interpretability of results
* Evaluation of retrieved context
* Overall system performance

Thus, using multiple documents aligns better with the objective of building and evaluating an effective RAG system.

## RAG Pipeline Design
- **Knowledge Base**: 7 entirely custom, specialized markdown reports derived from technical papers on Indian hydrology (e.g., glaciers, monsoons, groundwater, salinity).
- **Chunking Strategy**: `RecursiveCharacterTextSplitter` with `chunk_size=800` and `chunk_overlap=150`. This ensures that complex statistics (e.g., salinity metrics or rainfall data) are not cut off mid-sentence.
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` via Hugging Face for dense, specialized vector representation without heavy API costs.
- **Vector Database**: FAISS (Facebook AI Similarity Search) for low-latency retrieval.
- **Retrieval Strategy**: A semantic search retriever configured with `k=5` to pull multiple context windows across the 7 subtopic documents.
- **Inference Engine**: Dynamic Multi-LLM support (Groq/Llama-3, OpenAI/GPT-4, Gemini) allowing the user to select their preferred generation model via the Streamlit UI.

## Key Features
- **Niche Focus**: Includes data on the Himalayan "Third Pole," the Indo-Gangetic groundwater crisis, and Sunderbans salinity levels.
- **Premium Interface**: A "Data-Journalism" styled Streamlit app with glassmorphic UI elements and live metrics.
- **Quantitative Metrics**: Measures semantic alignment (Cosine-sim) and factual keyword overlap (ROUGE-L).
- **Transparency**: Shows retrieved context chunks for every query.

## Getting Started
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Yashwant-king/climate-water-rag-india.git
    cd climate-water-rag-india
    ```
2.  **Environment Setup**:
    - Create a `.env` file with your `GROQ_API_KEY`.
    - Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Data Ingestion**:
    - Build the vector database:
    ```bash
    python src/ingestion.py
    ```
4.  **Run Evaluation**:
    - Run the automated evaluation framework:
    ```bash
    cd evaluation
    python evaluate.py
    ```
5.  **Start the App**:
    ```bash
    streamlit run app.py
    ```

## Evaluation Framework & Methodology
The system is evaluated against a "Golden Dataset" of 15 high-precision Q&A pairs (found in `golden_dataset.csv`). We use an automated standard metrics script (`evaluation/evaluate.py`) alongside qualitative guidelines:

1.  **Quantitative Metrics**: 
    - **Cosine Similarity**: Uses `all-MiniLM-L6-v2` to measure semantic closeness between the LLM output and the golden dataset answer. An answer passes if the score is $\geq$ 0.75.
    - **ROUGE-L**: Measures the exact factual keyword match and structural overlap using Longest Common Subsequence formatting.
2.  **Qualitative Assessment**: A human-in-the-loop validation focuses on **Policy Relevance**, **Coherence**, and **Factual Integrity** using the Streamlit UI's exposed document chunks.

### Sample Evaluation Results
Based on a recent run of the evaluation suite (`report_generator.py`), the system achieved:
- **Total Questions Evaluated**: 15
- **Average Cosine Similarity**: ~0.916
- **Average ROUGE-L Score**: ~0.623
- **Pass Rate**: 100.0%

## Challenges Encountered and Lessons Learned
- **Framework Deprecation**: Navigating the transition from `langchain` generic classes to `langchain_community` and `langchain_huggingface` required meticulous dependency management to prevent breaking changes.
- **Chunk Crossover**: During evaluation, I found that standard chunking (e.g., splitting strictly by length) often severed crucial statistical linkages. Using `RecursiveCharacterTextSplitter` and bumping the size to 800 preserved "statistical semantic coherence" brilliantly.
- **Multi-Document Scaling**: Originally relying on 1-2 documents led to generic retrieval. Splitting the domain into 7 highly-specific documents forces the FAISS index to do actual "needle-in-a-haystack" work, which perfectly stress-tests the real capabilities of the RAG.

[Your Name] | AI Engineer Intern

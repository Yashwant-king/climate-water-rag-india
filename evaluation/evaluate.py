import os
import sys
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from rouge_score import rouge_scorer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Add src to the path
sys.path.append(os.path.abspath("../src"))
from generation import get_llm, get_rag_chain

load_dotenv(dotenv_path="../.env")

def evaluate_rag_system(csv_path, vector_store, llm):
    print("Initializing Evaluation...")
    data = pd.read_csv(csv_path)
    
    # Initialize Metrics
    model = SentenceTransformer('all-MiniLM-L6-v2')
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    # Initialize RAG Chain with the provided LLM
    chain = get_rag_chain(vector_store, llm)
    
    results = []
    
    print(f"Starting Evaluation on {len(data)} questions...")
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        question = row['question']
        expected_answer = row['expected_answer']
        
        # Generate Answer
        try:
            generated_answer = chain.invoke(question)
        except Exception as e:
            generated_answer = f"Error: {e}"
            
        # 1. Cosine Similarity (Semantic Match)
        embeddings = model.encode([expected_answer, generated_answer])
        cos_sim = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        # 2. ROUGE-L (Overlap)
        rouge_scores = scorer.score(expected_answer, generated_answer)
        rouge_l = rouge_scores['rougeL'].fmeasure
        
        # 3. Pass/Fail based on threshold
        is_pass = cos_sim >= 0.75
        
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "cosine_similarity": round(cos_sim, 4),
            "rouge_l_score": round(rouge_l, 4),
            "status": "PASS" if is_pass else "FAIL"
        })
        
    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    
    # Summary Table
    print("\n--- Evaluation Summary ---")
    print(f"Total Questions: {len(data)}")
    print(f"Average Cosine Similarity: {results_df['cosine_similarity'].mean():.4f}")
    print(f"Average ROUGE-L Score: {results_df['rouge_l_score'].mean():.4f}")
    print(f"Pass Rate: {(results_df['status'] == 'PASS').mean() * 100:.2f}%")
    
    return results_df

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    FAISS_PATH = "../data/faiss_index"
    
    if not os.path.exists(FAISS_PATH):
        print("FAISS index not found! Run src/ingestion.py first.")
    else:
        vector_store = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        
        # Default to Groq for evaluation (fast + free tier)
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            print("ERROR: GROQ_API_KEY not found in .env file!")
            print("Please set GROQ_API_KEY in the project root .env file.")
            sys.exit(1)
        
        llm = get_llm("Groq", groq_key, "llama-3.3-70b-versatile")
        print(f"Using LLM: Groq / llama-3.3-70b-versatile for evaluation.")
        evaluate_rag_system("golden_dataset.csv", vector_store, llm)

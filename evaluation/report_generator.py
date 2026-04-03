import pandas as pd
import json
import matplotlib.pyplot as plt

def generate_visual_report(csv_path="evaluation_results.csv"):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("Results file not found. Run evaluate.py first.")
        return

    # Visual 1: Summary Statistics
    summary = {
        "Total Questions": len(df),
        "Average Cosine Similarity": df["cosine_similarity"].mean(),
        "Average ROUGE-L": df["rouge_l_score"].mean(),
        "Pass Rate (%)": (df["status"] == "PASS").mean() * 100
    }
    
    with open("summary_report.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Visual 2: Metric Distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(df["cosine_similarity"], bins=10, color='skyblue', edgecolor='black')
    plt.title("Cosine Similarity Distribution")
    plt.xlabel("Score")
    
    plt.subplot(1, 2, 2)
    plt.hist(df["rouge_l_score"], bins=10, color='lightcoral', edgecolor='black')
    plt.title("ROUGE-L Score Distribution")
    plt.xlabel("Score")
    
    plt.tight_layout()
    plt.savefig("metrics_visualization.png")
    print("Report and visuals generated: summary_report.json, metrics_visualization.png")

if __name__ == "__main__":
    generate_visual_report()

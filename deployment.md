# Deployment Guide: Climate-Water RAG (India)

## Deployment Options

### 1. Streamlit Community Cloud (Easiest)
Deploy your AI Assistant directly from your GitHub repository for free.
1.  **Push to GitHub**: Make sure your `requirements.txt`, `app.py`, and `data/` are committed.
2.  **Streamlit Cloud**: Login to [share.streamlit.io](https://share.streamlit.io) and link your GitHub repo.
3.  **Secrets**: In the "Advanced Settings" of your Streamlit app, add your API keys:
    ```toml
    GROQ_API_KEY = "your-key"
    OPENAI_API_KEY = "your-key"
    GOOGLE_API_KEY = "your-key"
    ```

### 2. Hugging Face Spaces (Best for AI Models)
Hugging Face offers free hosting for Streamlit/Gradio apps.
1.  **Create a New Space**: Choose "Streamlit" as the SDK.
2.  **Upload Files**: Use Git or the HF UI to upload your project.
3.  **Variables**: Add your tokens to the "Settings > Repository Secrets" section.

### 3. Vercel / Render (Manual Deployment)
Best if you want to integrate this as a back-end service.
- **Backend API**: If using Groq/Llama 3, the latency will be very low on Render.
- **Port**: Ensure your app listens on the port provided by the platform (usually `$PORT`).

## Continuous Integration (CI)
The project includes an `evaluation/evaluate.py` script. You can set up a **GitHub Action** to run this evaluation every time you push new documents to the `data/` folder, ensuring your RAG system's accuracy hasn't "regressed."

## Maintenance
- **Data Updates**: Simply add more `.md` or `.pdf` files to `data/` and run `python src/ingestion.py` to update the knowledge base.
- **Model Upgrades**: Change the `model_name` in the Sidebar UI to test newer models (like GPT-5 or Llama 4) as they are released.

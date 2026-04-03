import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_llm(provider, api_key, model_name):
    """
    Returns the appropriate LLM instance based on user selection.
    """
    if provider == "Groq":
        return ChatGroq(api_key=api_key, model_name=model_name, temperature=0)
    elif provider == "OpenAI":
        return ChatOpenAI(api_key=api_key, model_name=model_name, temperature=0)
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(api_key=api_key, model=model_name, temperature=0)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")

def get_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    
    template = """You are an expert environmental scientist specializing in the impact of climate change on Indian water resources.
    Use the following retrieved context to answer the user's question accurately.
    Be factual, precise, and maintain a professional tone. If the context does not contain the answer, say you don't know based on the provided context.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

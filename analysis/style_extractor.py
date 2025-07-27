from ingestion.doc_ingestor import load_documents
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage, SystemMessage
from collections import Counter
import json
import time
import random    

STYLE_ANALYSIS_PROMPT = """
        You are a professional writing coach and language analyst.

        Your task is to deeply analyze a user's writing sample and return a detailed profile of their writing style, which includes their tone, sentence        structure, vocabulary, and rhetorical habits.

        You must return your output in JSON format, with the following keys:
        - tone: What is the emotional feel? (e.g., formal, optimistic, witty, bold, curious)
        - sentence_structure: How does the user construct sentences? (e.g., long and flowing, short and punchy, complex clauses)
        - vocabulary: What kind of words do they use? (e.g., advanced, plain, technical, figurative)
        - rhetorical_devices: What stylistic habits do they have? (e.g., use of metaphors, questions, humor, repetition)
        - summary: A short paragraph summarizing their overall voice
        
        Your goal is to help personalize future LLM outputs by recreating this voice accurately.
        You will be given a string to extract style from.
        Only return a raw JSON object. Do NOT wrap it in triple backticks or markdown formatting.
        """
def safe_invoke_with_backoff(llm, messages, max_retries=5):
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as e:
            if "ThrottlingException" in str(e):
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"⚠️ Throttled. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("❌ Exceeded max retries due to throttling.")

def grab_styles(chunks, llm):
    style_profiles = []
    for chunk in chunks:
        messages = [
            SystemMessage(content=STYLE_ANALYSIS_PROMPT),
            HumanMessage(content=f"Here is a writing sample: \n\n{chunk}")
        ]
        response = safe_invoke_with_backoff(llm, messages, 5)
        print("response successful")
        print("🔎 Raw LLM response:", repr(response.content))
        style_profiles.append(json.loads(response.content))
    return style_profiles

def aggregate_styles(profiles):
    def mode_or_concat(key):
        values = [p[key] for p in profiles]
        return Counter(values).most_common(1)[0][0]
    return {
        "tone": mode_or_concat("tone"),
        "sentence_structure": mode_or_concat("sentence_structure"),
        "vocabulary": mode_or_concat("vocabulary"),
        "rhetorical_devices": mode_or_concat("rhetorical_devices"),
        "summary": " ".join([p["summary"] for p in profiles])
    } 

def extract_full_style(chunks, llm):
    profiles = grab_styles(chunks, llm)
    return aggregate_styles(profiles)
    
    


from langchain.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

style_system_prompt = "" 

def extract_style_summary(docs, llm) -> dict:
    prompt = PromptTemplate.from_template(
        """
        You are a professional writing coach and language analyst.

        Your task is to deeply analyze a user's writing sample and return a detailed profile of their writing style, which includes their tone, sentence        structure, vocabulary, and rhetorical habits.

        You must return your output in JSON format, with the following keys:
        - tone: What is the emotional feel? (e.g., formal, optimistic, witty, bold, curious)
        - sentence_structure: How does the user construct sentences? (e.g., long and flowing, short and punchy, complex clauses)
        - vocabulary: What kind of words do they use? (e.g., advanced, plain, technical, figurative)
        - rhetorical_devices: What stylistic habits do they have? (e.g., use of metaphors, questions, humor, repetition)
        - summary: A short paragraph summarizing their overall voice
        
        Your goal is to help personalize future LLM outputs by recreating this voice accurately.

        Here is the sample: 
        {content}
        Do not include commentary or extra text — only return a clean JSON object.
        """
    )
    chain = LLMChain(llm=llm, prompt=prompt) 
    result = chain.run()

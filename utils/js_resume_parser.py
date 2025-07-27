from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
import time

def parse_resume(resume_text: str, llm) -> dict:
    prompt= PromptTemplate.from_template("""
    From the user's resume, extract their skills as a list, project names as a list, previous experiences as a list, and generate a summary as a string.    Respond
    in a json format in which the information is organized with these keys
        - Full name 
        - Most Relevant Experience (2-3 Bullet Points)
        - Key Technical Skills
        - Soft Skills
        - Summary (2-3 sentences)
    Resume:  {resume_text}
    Respond in JSON format.
    """)

    try:
        print("Parsing Resume and Sending Prompt...")
        chain = prompt | llm
        result = chain.invoke({
            "resume_text": resume_text,
        }
                              )
        time.sleep(1.5)
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse resume: {e}")

def parse_job_description(jd_text: str, llm) -> dict:
    prompt = PromptTemplate.from_template(""" From this Job description, extract the exact role title as a string, 
    the requirements they outline as a list, any preferred qualifications as a list, 
    and the overall vibe of the job and place it into this format as a concise string
    Use the following format:
        - Job Title
        - Company
        - Top 3 Responsibilities
        - Desired Skills
        - Culture / Values
        - Overall vibe

    Job Description: 
    {jd_text}
    Respond in JSON format.
    """)
    try:
        print("Parsing J*b Description and Sending Prompt")
        chain = prompt | llm 
        result = chain.invoke({
            "jd_text": jd_text,
        })
        time.sleep(1.5)
        return result
    except Exception as e:
        raise ValueError(f"Failed to parse j*b description: {e}")

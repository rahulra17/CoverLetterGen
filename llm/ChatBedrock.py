from langchain_aws import ChatBedrock
import os
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import time
import random

ClaudeSonnet = ChatBedrock (
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0",
    region_name = os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"), 
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
)

ClaudeHaiku = ChatBedrock(
    model_id = "us.anthropic.claude-3-haiku-20240307-v1:0",
    region_name = os.getenv("AWS_DEFAULT_REGION"),
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID"), 
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY"),
)

messages = [
    (
        "system",
        """

You are a professional writing assistant that specializes in crafting personalized, voice-aligned cover letters for job applications.

You will receive:
- A writing style profile that reflects the applicant’s tone, word choice, and sentence structure
- A parsed resume with experiences, skills, and accomplishments
- A parsed job description with key responsibilities and values the company is looking for

Your task is to write a cover letter that:
- Sounds like the applicant based on their writing style profile
- Clearly aligns the applicant’s past experiences and values with what the company is looking for
- Uses a confident, sharp, and human tone — not robotic or overly generic
- Includes specific details rather than clichés or fluff
- Follows proper formatting and flows naturally
- Include one or two run-on sentences and one or two grammatical errors to make it sound more human. It should score a 92 on grammarly

Structure the cover letter in 3–5 paragraphs:
1. A strong, personal hook that shows interest in the company/role
2. A section highlighting the most relevant experience, tailored to the job
3. Optional: a secondary story or experience that reinforces culture fit
4. A brief conclusion reaffirming interest and inviting next steps

Do not copy the resume — synthesize and tell a compelling story.

Always write in the applicant’s voice. Maintain consistency with their tone, cadence, and rhetorical style.

Return only the final cover letter text, no additional commentary.
"""
    ),
    ("human", 
     """
     Write a personalized cover letter using the following:
     Resume: {resume_json}
     Job Description: {jd_json}
     Writing Style: {style_profile}
     """)
]

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

def safe_chain_invoke_with_backoff(chain, resume, jd, style, max_retries=5):
    for attempt in range(max_retries):
        try:
            return chain.invoke({
                "resume_json": resume,
                "jd_json": jd,
                "style_profile": style,
            })
        except Exception as e:
            if "ThrottlingException" in str(e):
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"⚠️ Throttled. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                raise e
    raise RuntimeError("❌ Exceeded max retries due to throttling.")

   

def invoke_claude(style: dict, resume: dict, jd: dict):
    prompt = ChatPromptTemplate.from_messages(messages) 
    chain = prompt | ClaudeSonnet
    result = safe_chain_invoke_with_backoff(chain, resume, jd, style).dict()
    return result

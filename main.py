from utils.js_resume_parser import parse_job_description, parse_resume
from llm.ChatBedrock import invoke_claude, ClaudeSonnet
from analysis.style_extractor import extract_full_style
from ingestion.doc_ingestor import load_documents, chunk_documents

resume_txt = "" 
with open("data/resume.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

jd_text = ""
with open("data/job_description.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()

resume = parse_resume(resume_txt, ClaudeSonnet)

jd = parse_job_description(jd_text, ClaudeSonnet)

docs = load_documents("./data/samples/")

chunks = chunk_documents(docs)

style_profile = extract_full_style(chunks, ClaudeSonnet)

final_result = invoke_claude(style_profile, resume, jd)

print(final_result)

from utils.js_resume_parser import parse_job_description, parse_resume
from llm.ChatBedrock import invoke_claude, ClaudeSonnet, ClaudeHaiku
from analysis.style_extractor import extract_full_style
from ingestion.doc_ingestor import load_documents, chunk_documents, batch_chunks

resume_txt = "" 
with open("data/resume.txt", "r", encoding="utf-8") as f:
    resume_txt = f.read()

print(resume_txt)
jd_text = ""
with open("data/job_description.txt", "r", encoding="utf-8") as f:
    jd_text = f.read()
print(jd_text)

resume = parse_resume(resume_txt, ClaudeHaiku)

jd = parse_job_description(jd_text, ClaudeHaiku)

print("Resume and JD")

print(resume)

print(jd)

docs = load_documents("./data/samples/")

chunks = chunk_documents(docs)

batch_chunks = batch_chunks(chunks)
# leaner model to fit throughput constraints
style_profile = extract_full_style(batch_chunks, ClaudeHaiku)

print("made it past the penultiamte hurdle")

final_result = invoke_claude(style_profile, resume, jd)

print(final_result)

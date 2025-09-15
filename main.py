from utils.js_resume_parser import parse_job_description, parse_resume
from llm.ChatBedrock import invoke_claude, ClaudeSonnet, ClaudeHaiku
from analysis.style_extractor import extract_full_style
from ingestion.doc_ingestor import load_documents, chunk_documents, batch_chunks
import streamlit as st 
from langchain_core.documents import Document

st.set_page_config(page_title="CoverLetterGen", layout="wide")
st.title("CoverLetterGen")
st.subheader("Made by Rahul Ramineni")

try:
    haiku = ClaudeHaiku
    sonnet = ClaudeSonnet
except Exception as e:
    st.error(f"Error Loading LLM: {e}")

st.sidebar.title("Options!")
show_sources = st.sidebar.checkbox("Show Source Chunks", value=True)

if "writing_samples" not in st.session_state:
    st.session_state.writing_samples = []

st.title("Submit your writing samples")

sample_input = st.text_area("Enter a writing sample: ", key="Sample_box")

if st.button("Submit Sample"):
    with st.spinner("Submitting..."):
        if sample_input.strip():
            st.session_state.writing_samples.append(sample_input.strip())
            st.success(f"Sample #{len(st.session_state.writing_samples)} added!")
            st.session_state.sample_box = ""
        else:
            st.warning("Enter some text bro so we know how to write it come on now")

if st.session_state.writing_samples:
    st.subheader("Submitted Samples:")
    for i, sample in enumerate(st.session_state.writing_samples, 1):
        st.markdown(f"**Sample {i}:** {sample[:100]}{'...' if len(sample) > 100 else ''}")

def load_from_session_state(writing_samples: list):
    return[Document(page_content=sample) for sample in writing_samples]

if "style_profile" not in st.session_state:
    st.session_state.style_profile = {}

if len(st.session_state.writing_samples) > 0:
    if st.button("analyze writing styles"):
        with st.spinner("Loading samples..."):
            try:
                docs = load_from_session_state(st.session_state.writing_samples)
            except Exception as e:
                st.error(f"Oh nooo my fault there's an error: {e}")
            with st.spinner("Chunking it up (chunky monkey)..."):
                try:
                    chunks = chunk_documents(docs)
                except Exception as e:
                    st.error(f"Oh noooo I couldn't chunk it: {e}")
                with st.spinner("Batching_Chunks..."):
                    try:
                        batch_chunks=batch_chunks(chunks)
                    except Exception as e:
                        st.error(f"Oh noooo I couldn't batch the chunks: {e}")
                    with st.spinner("Extracting Style now! Patience my friend almost there..."):
                        try:
                            style_profile=extract_full_style(batch_chunks, haiku)
                        except Exception as e:
                            st.error(f"Oh nooo I couldn't extract style: {e}")
                        st.session_state.style_profile = style_profile
                        st.success("Analyzed Documents and now I know how you write!")


if "resume_txt" not in st.session_state:
    st.session_state.resume_txt = ""
st.title("Give us your Resume plsplspslspl")

resume_input = st.text_area("Enter your resume(j copy and paste it bro)", key="Resume_box")
if st.button("Submit Resume"):
    with st.spinner("Logging Resume"):
        if resume_input.strip():
            st.session_state.resume_txt=resume_input
            st.success("Resume added!")
        else:
            st.warning("put your damn resume in!")

if st.session_state.resume_txt:
    st.subheader("Submitted Resume:")
    st.markdown(f"***Resume: {resume_input[:100]}{'...' if len(resume_input) > 100 else ''}")

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
st.title("PLEASE give us a j*b description here we beg")

jd_input = st.text_area("Enter the job description (j copy and paste it bro)", key="Job_description_box")
if st.button("Submit Job Descrption"):
    with st.spinner("Oh wow this job looks cool"):
        st.subheader(f"Here is the jd input: {jd_input}")
        st.session_state.jd_text = jd_input
        st.success("Job Description added!")

if st.session_state.jd_text:
    st.subheader("Submitted Job description!")
    st.markdown(f"***Job Description: {jd_input[:100]}{'...' if len(jd_input) > 100 else ''}")

if "cover_letter" not in st.session_state:
    st.session_state.cover_letter = {}

if st.session_state.style_profile and st.session_state.resume_txt and st.session_state.jd_text:
    if st.button("Create Cover Letter"):
        with st.spinner("Asking our magician friend to make this for you. Please hold"):
            try:
                resume_json = parse_resume(st.session_state.resume_txt, haiku)
                jd_json = parse_job_description(st.session_state.jd_text, haiku)
                result = invoke_claude(st.session_state.style_profile, resume_json, jd_json)
                print("LOOOOOK HERERERERE")
                print(type(result))
            except Exception as e:
                st.error("Something happened with our friend so he can't help us :(")
            st.markdown(f"***Cover Letter: {result}")
            st.session_state.cover_letter = result.get("content")
            st.success("Cover Letter Generated!")

if st.session_state.cover_letter:
    if st.button("Show Cover Letter"):
        st.markdown(f"{st.session_state.cover_letter}")


from utils.js_resume_parser import parse_job_description, parse_resume
from llm.ChatBedrock import invoke_claude, ClaudeSonnet, ClaudeHaiku
from analysis.style_extractor import extract_full_style
from ingestion.doc_ingestor import load_documents, chunk_documents, batch_chunks
import streamlit as st 
from langchain_core.documents import Document
import time

# Page configuration
st.set_page_config(
    page_title="CoverLetterGen", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def init_session_state():
    if "writing_samples" not in st.session_state:
        st.session_state.writing_samples = []
    if "style_profile" not in st.session_state:
        st.session_state.style_profile = {}
    if "resume_txt" not in st.session_state:
        st.session_state.resume_txt = ""
    if "jd_text" not in st.session_state:
        st.session_state.jd_text = ""
    if "cover_letter" not in st.session_state:
        st.session_state.cover_letter = ""
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1

# Initialize LLM models
try:
    haiku = ClaudeHaiku
    sonnet = ClaudeSonnet
except Exception as e:
    st.error(f"Error Loading LLM: {e}")

# Helper functions
def load_from_session_state(writing_samples: list):
    return [Document(page_content=sample) for sample in writing_samples]

def show_progress_bar():
    progress = st.session_state.current_page / 4
    st.progress(progress)
    st.caption(f"Step {st.session_state.current_page} of 4")

def show_navigation():
    # Create a container that will be positioned at the bottom
    with st.container():
        st.markdown("---")  # Add a separator line
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.session_state.current_page > 1:
                if st.button("← Previous", key="prev_btn", use_container_width=True):
                    st.session_state.current_page -= 1
                    st.rerun()
        
    with col2:
        st.write(f"**Page {st.session_state.current_page} of 4**")
    
    with col3:
        if st.session_state.current_page < 4:
            if st.button("Next →", key="next_btn", type="primary", use_container_width=True):
                st.session_state.current_page += 1
                st.rerun()

# Page 1: Writing Samples Collection
def page1_writing_samples():
    st.title("📝 Step 1: Share Your Writing Style")
    st.markdown("Help us understand your unique voice by sharing samples of your writing.")
    
    st.info("💡 **Tip:** Share 2-3 samples of your best writing - emails, essays, reports, or any content that represents your style.")
    
    with st.container():
        st.subheader("Add Writing Sample")
        
        # Use a form to handle the input and clearing
        with st.form("sample_form", clear_on_submit=True):
            sample_input = st.text_area(
                "Paste your writing sample here:", 
                height=200,
                placeholder="Enter any piece of writing that represents your style..."
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                submitted = st.form_submit_button("Add Sample", type="primary")
            
            with col2:
                if st.session_state.writing_samples and st.form_submit_button("Clear All Samples"):
                    st.session_state.writing_samples = []
                    st.rerun()
        
        # Handle form submission
        if submitted:
            if sample_input.strip():
                # Check for duplicates
                if sample_input.strip() in st.session_state.writing_samples:
                    st.warning("⚠️ This sample has already been added. Please enter a different writing sample.")
                else:
                    st.session_state.writing_samples.append(sample_input.strip())
                    st.success(f"✅ Sample #{len(st.session_state.writing_samples)} added!")
            else:
                st.warning("Please enter some text to add as a sample.")
    
    # Display submitted samples
    if st.session_state.writing_samples:
        st.subheader("📋 Your Writing Samples")
        
        # Add delete functionality for each sample
        for i, sample in enumerate(st.session_state.writing_samples, 1):
            with st.expander(f"Sample {i} ({len(sample)} characters)"):
                # Sample content
                st.write(sample)
                
                # Delete button for this specific sample
                col1, col2, col3 = st.columns([1, 1, 4])
                
                with col1:
                    if st.button(f"🗑️ Delete", key=f"delete_sample_{i}", type="secondary"):
                        # Remove the sample from the list
                        st.session_state.writing_samples.pop(i-1)
                        # Clear style profile since samples changed
                        if st.session_state.style_profile:
                            st.session_state.style_profile = {}
                        st.success(f"✅ Sample {i} deleted! Style analysis will need to be re-run.")
                        st.rerun()
                
                with col2:
                    if st.button(f"📋 Copy", key=f"copy_sample_{i}"):
                        st.code(sample, language=None)
                        st.success("✅ Sample copied to clipboard!")
                
                with col3:
                    st.write("")  # Empty space for alignment
        
        # Style analysis
        if st.button("🔍 Analyze My Writing Style", type="primary"):
            with st.spinner("Analyzing your writing style... This may take a moment."):
                try:
                    docs = load_from_session_state(st.session_state.writing_samples)
                    chunks = chunk_documents(docs)
                    batch_chunks_list = list(batch_chunks(chunks))
                    style_profile = extract_full_style(batch_chunks_list, haiku)
                    st.session_state.style_profile = style_profile
                    st.success("🎉 Writing style analyzed successfully!")
                    
                    # Show style summary with professional UI
                    st.markdown("### 📊 Your Writing Style Profile")
                    
                    # Create a professional style profile display
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🎭 **Tone & Voice**")
                        tone_color = {
                            "formal": "🔵",
                            "casual": "🟢", 
                            "professional": "🔵",
                            "conversational": "🟡",
                            "witty": "🟣",
                            "confident": "🟠",
                            "optimistic": "🟡"
                        }.get(style_profile.get("tone", "").lower(), "⚪")
                        
                        st.markdown(f"**Tone:** {tone_color} {style_profile.get('tone', 'N/A')}")
                        
                        st.markdown("#### 📝 **Writing Structure**")
                        st.markdown(f"**Style:** {style_profile.get('sentence_structure', 'N/A')}")
                        
                    with col2:
                        st.markdown("#### 📚 **Vocabulary & Language**")
                        vocab_color = {
                            "advanced": "🔴",
                            "technical": "🔵",
                            "plain": "🟢",
                            "figurative": "🟣",
                            "professional": "🟢"
                        }.get(style_profile.get("vocabulary", "").lower(), "⚪")
                        
                        st.markdown(f"**Level:** {vocab_color} {style_profile.get('vocabulary', 'N/A')}")
                        
                        st.markdown("#### ✨ **Rhetorical Style**")
                        st.markdown(f"**Devices:** {style_profile.get('rhetorical_devices', 'N/A')}")
                    
                    # Summary section
                    st.markdown("#### 📝 **Style Summary**")
                    st.info(f"💬 *{style_profile.get('summary', 'No summary available')}*")
                    
                    # Visual indicators
                    st.markdown("---")
                    st.markdown("#### 🎯 **Style Characteristics**")
                    
                    # Create visual indicators for style elements
                    style_elements = [
                        ("Tone", style_profile.get("tone", "Unknown")),
                        ("Structure", style_profile.get("sentence_structure", "Unknown")),
                        ("Vocabulary", style_profile.get("vocabulary", "Unknown")),
                        ("Rhetorical Devices", style_profile.get("rhetorical_devices", "Unknown"))
                    ]
                    
                    for element, value in style_elements:
                        st.markdown(f"**{element}:** `{value}`")
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing writing style: {e}")
    else:
        st.info("Add at least one writing sample to continue.")

# Page 2: Resume & Job Description
def page2_resume_job():
    st.title("📄 Step 2: Resume & Job Description")
    st.markdown("Upload your resume and the job description you're applying for.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Your Resume")
        resume_input = st.text_area(
            "Paste your resume here:",
            key="resume_input",
            height=300,
            placeholder="Copy and paste your resume content..."
        )
        
        if st.button("Save Resume", key="save_resume"):
            if resume_input.strip():
                st.session_state.resume_txt = resume_input
                st.success("✅ Resume saved!")
            else:
                st.warning("Please enter your resume content.")
        
        if st.session_state.resume_txt:
            st.success("✅ Resume loaded")
            with st.expander("Preview Resume"):
                st.text(st.session_state.resume_txt[:500] + "..." if len(st.session_state.resume_txt) > 500 else st.session_state.resume_txt)
    
    with col2:
        st.subheader("💼 Job Description")
        jd_input = st.text_area(
            "Paste the job description here:",
            key="jd_input", 
            height=300,
            placeholder="Copy and paste the job description..."
        )
        
        if st.button("Save Job Description", key="save_jd"):
            if jd_input.strip():
                st.session_state.jd_text = jd_input
                st.success("✅ Job description saved!")
            else:
                st.warning("Please enter the job description.")
        
        if st.session_state.jd_text:
            st.success("✅ Job description loaded")
            with st.expander("Preview Job Description"):
                st.text(st.session_state.jd_text[:500] + "..." if len(st.session_state.jd_text) > 500 else st.session_state.jd_text)
    
    # Validation
    if st.session_state.resume_txt and st.session_state.jd_text:
        st.success("🎉 Ready to generate your cover letter!")
    else:
        missing = []
        if not st.session_state.resume_txt:
            missing.append("resume")
        if not st.session_state.jd_text:
            missing.append("job description")
        st.warning(f"Please provide your {' and '.join(missing)} to continue.")

# Page 3: Cover Letter Generation
def page3_generation():
    st.title("✨ Step 3: Generate Your Cover Letter")
    st.markdown("Create a personalized cover letter that matches your writing style.")
    
    # Check prerequisites
    if not st.session_state.style_profile:
        st.error("❌ Please complete Step 1 (Writing Samples) first.")
        return
    
    if not st.session_state.resume_txt:
        st.error("❌ Please complete Step 2 (Resume) first.")
        return
        
    if not st.session_state.jd_text:
        st.error("❌ Please complete Step 2 (Job Description) first.")
        return
    
    st.success("✅ All prerequisites completed! Ready to generate.")
    
    # Generation section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎯 Generation Settings")
        
        # Show style profile summary with professional UI
        st.markdown("#### 🎭 **Your Writing Style**")
        
        # Compact style profile display
        style_profile = st.session_state.style_profile
        
        # Tone indicator
        tone_emoji = {
            "formal": "🔵", "casual": "🟢", "professional": "🔵", 
            "conversational": "🟡", "witty": "🟣", "confident": "🟠", 
            "optimistic": "🟡"
        }.get(style_profile.get("tone", "").lower(), "⚪")
        
        st.markdown(f"**Tone:** {tone_emoji} {style_profile.get('tone', 'N/A')}")
        st.markdown(f"**Structure:** {style_profile.get('sentence_structure', 'N/A')}")
        st.markdown(f"**Vocabulary:** {style_profile.get('vocabulary', 'N/A')}")
        
        
        # Generation button
        if st.button("🚀 Generate Cover Letter", type="primary", use_container_width=True):
            with st.spinner("Creating your personalized cover letter..."):
                try:
                    # Parse resume and job description
                    with st.spinner("Parsing resume..."):
                        resume_json = parse_resume(st.session_state.resume_txt, haiku)
                    
                    with st.spinner("Parsing job description..."):
                        jd_json = parse_job_description(st.session_state.jd_text, haiku)
                    
                    # Generate cover letter
                    with st.spinner("Generating cover letter..."):
                        result = invoke_claude(st.session_state.style_profile, resume_json, jd_json)
                        st.session_state.cover_letter = result.get("content", "")
                    
                    st.success("🎉 Cover letter generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Error generating cover letter: {e}")
    
    with col2:
        st.subheader("📄 Your Cover Letter")
        
        if st.session_state.cover_letter:
            # Create a professional cover letter display box
            st.markdown("---")
            
            # Cover letter content in a styled container
            with st.container():
                st.markdown("""
                <div style="
                    background-color: #f8f9fa;
                    border: 2px solid #e9ecef;
                    border-radius: 10px;
                    padding: 20px;
                    margin: 10px 0;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    font-family: 'Georgia', serif;
                    line-height: 1.6;
                    white-space: pre-wrap;
                ">
                """, unsafe_allow_html=True)
                
                # Display the cover letter content
                st.markdown(st.session_state.cover_letter)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Action buttons in a more prominent layout
            st.markdown("#### 📋 Actions")
            col_copy, col_download = st.columns(2)
            
            with col_copy:
                if st.button("📋 Copy to Clipboard", use_container_width=True, type="secondary"):
                    st.code(st.session_state.cover_letter, language=None)
                    st.success("✅ Cover letter copied to clipboard!")
            
            with col_download:
                if st.button("💾 Download as Text", use_container_width=True, type="secondary"):
                    st.download_button(
                        label="📄 Download Cover Letter",
                        data=st.session_state.cover_letter,
                        file_name="cover_letter.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
            # Additional info
            st.info("💡 **Tip:** You can copy the text above or download it as a .txt file for easy use in your applications.")
            
        else:
            # Empty state with better styling
            st.markdown("---")
            with st.container():
                st.markdown("""
                <div style="
                    background-color: #f8f9fa;
                    border: 2px dashed #dee2e6;
                    border-radius: 10px;
                    padding: 40px 20px;
                    margin: 10px 0;
                    text-align: center;
                    color: #6c757d;
                ">
                    <h4 style="color: #495057; margin-bottom: 10px;">📝 Ready to Generate</h4>
                    <p>Click the "Generate Cover Letter" button to create your personalized cover letter.</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("---")
    
    # Full-width detailed style analysis expander
    if st.session_state.style_profile:
        st.markdown("---")
        with st.expander("📊 Detailed Style Analysis + Cover Letter", expanded=False):
            st.markdown("#### 🎯 **Complete Style Profile**")
            
            # Use full width for better display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Tone & Voice**")
                st.code(style_profile.get('tone', 'N/A'))
                
                st.markdown("**Sentence Structure**")
                st.code(style_profile.get('sentence_structure', 'N/A'))
            
            with col2:
                st.markdown("**Vocabulary Level**")
                st.code(style_profile.get('vocabulary', 'N/A'))
                
                st.markdown("**Rhetorical Devices**")
                st.code(style_profile.get('rhetorical_devices', 'N/A'))
            
            with col3:
                st.markdown("**Style Summary**")
                st.info(style_profile.get('summary', 'No summary available'))
            
            # Cover Letter section in the detailed view
            if st.session_state.cover_letter:
                st.markdown("---")
                st.markdown("#### 📄 **Generated Cover Letter**")
                
                # Cover letter in a styled container within the expander
                st.markdown("""
                <div style="
                    background-color: #ffffff;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px 0;
                    font-family: 'Georgia', serif;
                    line-height: 1.5;
                    white-space: pre-wrap;
                    max-height: 400px;
                    overflow-y: auto;
                ">
                """, unsafe_allow_html=True)
                
                st.markdown(st.session_state.cover_letter)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Action buttons within the expander
                col_copy, col_download = st.columns(2)
                
                with col_copy:
                    if st.button("📋 Copy Cover Letter", key="copy_detailed", use_container_width=True):
                        st.code(st.session_state.cover_letter, language=None)
                        st.success("✅ Cover letter copied!")
                
                with col_download:
                    if st.button("💾 Download Cover Letter", key="download_detailed", use_container_width=True):
                        st.download_button(
                            label="📄 Download",
                            data=st.session_state.cover_letter,
                            file_name="cover_letter.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
            else:
                st.info("💡 Generate a cover letter first to see it in the detailed view.")

# Page 4: Final Cover Letter Display
def page4_final_cover_letter():
    st.title("📄 Step 4: Your Final Cover Letter")
    st.markdown("Review and finalize your personalized cover letter.")
    
    # Check prerequisites
    if not st.session_state.cover_letter:
        st.error("❌ Please complete Step 3 (Generate Cover Letter) first.")
        return
    
    st.success("✅ Cover letter ready for review!")
    
    # Main cover letter display
    st.markdown("---")
    
    # Cover letter in a professional document-style container
    with st.container():
        st.markdown("""
        <div style="
            background-color: #ffffff;
            border: 3px solid #2c3e50;
            border-radius: 15px;
            padding: 40px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            font-family: 'Georgia', serif;
            line-height: 1.8;
            white-space: pre-wrap;
            font-size: 16px;
            color: #2c3e50;
        ">
        """, unsafe_allow_html=True)
        
        # Display the cover letter content
        st.markdown(st.session_state.cover_letter)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Action buttons
    st.markdown("#### 📋 Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📋 Copy to Clipboard", type="primary", use_container_width=True):
            st.code(st.session_state.cover_letter, language=None)
            st.success("✅ Cover letter copied to clipboard!")
    
    with col2:
        if st.button("💾 Download as Text", type="secondary", use_container_width=True):
            st.download_button(
                label="📄 Download Cover Letter",
                data=st.session_state.cover_letter,
                file_name="cover_letter.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    with col3:
        if st.button("🔄 Generate New Version", type="secondary", use_container_width=True):
            st.session_state.cover_letter = ""
            st.success("✅ Cover letter cleared. Go back to Step 3 to generate a new version.")
            st.rerun()
    
    # Style profile summary
    if st.session_state.style_profile:
        st.markdown("---")
        st.markdown("#### 🎭 **Your Writing Style Used**")
        
        style_profile = st.session_state.style_profile
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tone", style_profile.get('tone', 'N/A'))
        with col2:
            st.metric("Structure", style_profile.get('sentence_structure', 'N/A'))
        with col3:
            st.metric("Vocabulary", style_profile.get('vocabulary', 'N/A'))
        with col4:
            st.metric("Devices", style_profile.get('rhetorical_devices', 'N/A'))

# Main app
def main():
    init_session_state()
    
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; margin-bottom: 0.5rem;">📝 CoverLetterGen</h1>
        <p style="color: #666; font-size: 1.1rem;">AI-Powered Personalized Cover Letters</p>
        <p style="color: #888; font-size: 0.9rem;">Made by Rahul Ramineni</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Progress bar
    show_progress_bar()
    
    # Page content
    if st.session_state.current_page == 1:
        page1_writing_samples()
    elif st.session_state.current_page == 2:
        page2_resume_job()
    elif st.session_state.current_page == 3:
        page3_generation()
    elif st.session_state.current_page == 4:
        page4_final_cover_letter()
    
    # Navigation at the bottom
    show_navigation()
    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 📊 Progress")
        show_progress_bar()
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This tool analyzes your writing style and creates personalized cover letters that sound authentically like you.
        
        **Steps:**
        1. **Writing Samples** - Share examples of your writing
        2. **Resume & Job** - Upload your resume and job description  
        3. **Generate** - Create your personalized cover letter
        4. **Final Review** - Review and finalize your cover letter
        """)

if __name__ == "__main__":
    main()
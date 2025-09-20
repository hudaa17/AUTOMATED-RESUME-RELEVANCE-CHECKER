def main():
    st.set_page_config(page_title="INNOMATICS RESEARCH LABS - Resume Evaluator", page_icon="🧠", layout="wide")
    
    st.markdown("""
        <div style="background-color:#0A74DA;padding:20px;border-radius:10px">
            <h1 style="color:white;text-align:center;">INNOMATICS RESEARCH LABS</h1>
            <h3 style="color:white;text-align:center;">Automated Resume Relevance Check System (LLM-based)</h3>
        </div>
    """, unsafe_allow_html=True)

    db = DatabaseManager()
    processor = DocumentProcessor()
    parser = ResumeParser()
    evaluator = RelevanceEvaluatorLLM()

    st.sidebar.title("Menu")
    menu = ["Upload Job Description", "Upload Resumes", "Evaluate Resumes", "View Results"]
    choice = st.sidebar.radio("", menu)

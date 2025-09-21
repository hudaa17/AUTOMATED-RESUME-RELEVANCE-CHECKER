"""
Main Streamlit application class
"""

import streamlit as st
import pandas as pd
import sqlite3
import json
import traceback
from datetime import datetime

from .database.manager import DatabaseManager
from .document.processor import DocumentProcessor
from .parsers.resume_parser import ResumeParser
from .parsers.job_parser import JobDescriptionParser
from .evaluators.relevance_evaluator import RelevanceEvaluator
from .utils.logger import setup_logger
from .utils.system_check import check_system_dependencies

logger = setup_logger(__name__)

class ResumeRelevanceApp:
    """Main Streamlit application with robust error handling"""
    
    def __init__(self):
        try:
            self.db_manager = DatabaseManager()
            self.doc_processor = DocumentProcessor()
            self.resume_parser = ResumeParser()
            self.jd_parser = JobDescriptionParser()
            self.evaluator = RelevanceEvaluator()
        except Exception as e:
            st.error(f"Application initialization failed: {e}")
            st.error(traceback.format_exc())
            st.stop()
    
    def run(self):
        """Run the application with error handling"""
        try:
            st.title("📄 AI-Powered Resume Relevance System")
            st.markdown("**Innomatics Research Labs** - Placement Analytics Dashboard")
            
            # Show system status
            self._show_system_status()
            
            # Navigation
            st.sidebar.title("Navigation")
            page = st.sidebar.selectbox(
                "Select Page",
                ["Job Management", "Resume Processing", "View Results", "System Info"]
            )
            
            if page == "Job Management":
                self._job_management_page()
            elif page == "Resume Processing":
                self._resume_processing_page()
            elif page == "View Results":
                self._results_page()
            elif page == "System Info":
                self._system_info_page()
                
        except Exception as e:
            st.error(f"Application error: {e}")
            st.error("Please refresh the page and try again.")
            logger.error(f"Application error: {e}")
    
    def _show_system_status(self):
        """Show system component status"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Status")
        
        # Component status
        components = check_system_dependencies()
        
        for component, status in components.items():
            st.sidebar.write(f"{component}: {status}")
    
    def _job_management_page(self):
        """Job management page"""
        st.header("📋 Job Management")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Create New Job Posting")
            
            job_title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
            location = st.selectbox("Location", ["Hyderabad", "Bangalore", "Pune", "Delhi NCR"])
            role_level = st.selectbox("Role Level", ["Entry Level", "Mid-level", "Senior"])
            
            jd_text = st.text_area("Job Description", height=200,
                placeholder="Paste complete job description...")
            
            if st.button("Create Job Posting", type="primary"):
                if job_title and jd_text:
                    try:
                        job = self.jd_parser.parse_job_description(jd_text, job_title, location)
                        self.db_manager.save_job_requirement(job)
                        
                        st.success("Job posting created successfully!")
                        
                        # Show parsed information
                        with st.expander("Parsed Job Information"):
                            st.write(f"**Job ID:** {job.job_id}")
                            st.write(f"**Must-have Skills:** {', '.join(job.must_have_skills)}")
                            st.write(f"**Good-to-have Skills:** {', '.join(job.good_to_have_skills)}")
                            st.write(f"**Required Experience:** {job.experience_years} years")
                            st.write(f"**Qualifications:** {', '.join(job.qualifications)}")
                        
                    except Exception as e:
                        st.error(f"Error creating job: {e}")
                        logger.error(f"Error creating job: {e}")
                else:
                    st.warning("Please fill in required fields")
        
        with col2:
            st.subheader("Recent Job Postings")
            try:
                conn = sqlite3.connect(self.db_manager.db_path)
                df = pd.read_sql_query(
                    "SELECT job_id, title, location, created_at FROM job_requirements ORDER BY created_at DESC LIMIT 10",
                    conn
                )
                conn.close()
                
                if not df.empty:
                    for _, job in df.iterrows():
                        with st.container():
                            st.write(f"**{job['title']}**")
                            st.write(f"Location: {job['location']}")
                            st.write(f"Created: {job['created_at'][:10]}")
                            st.markdown("---")
                else:
                    st.info("No job postings found")
                    
            except Exception as e:
                st.error(f"Error loading jobs: {e}")
                logger.error(f"Error loading jobs: {e}")
    
    def _resume_processing_page(self):
        """Resume processing page"""
        st.header("📄 Resume Processing")
        
        # Get available jobs
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            jobs_df = pd.read_sql_query(
                "SELECT job_id, title, location FROM job_requirements ORDER BY created_at DESC",
                conn
            )
            conn.close()
            
            if jobs_df.empty:
                st.warning("No job descriptions found. Please create a job posting first.")
                return
                
        except Exception as e:
            st.error(f"Error loading jobs: {e}")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload and Process Resumes")
            
            # Job selection
            job_options = [f"{row['title']} - {row['location']}" for _, row in jobs_df.iterrows()]
            selected_job_idx = st.selectbox("Select Job Posting", 
                                           range(len(job_options)), 
                                           format_func=lambda x: job_options[x])
            selected_job_id = jobs_df.iloc[selected_job_idx]['job_id']
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload Resume Files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT"
            )
            
            if uploaded_files:
                st.info(f"{len(uploaded_files)} files selected")
                
                if st.button("Process Resumes", type="primary"):
                    self._process_resumes(uploaded_files, selected_job_id)
        
        with col2:
            st.subheader("Processing Statistics")
            try:
                conn = sqlite3.connect(self.db_manager.db_path)
                
                total_resumes = conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
                st.metric("Total Resumes", total_resumes)
                
                total_evaluations = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
                st.metric("Total Evaluations", total_evaluations)
                
                if total_evaluations > 0:
                    avg_score = conn.execute("SELECT AVG(relevance_score) FROM evaluations").fetchone()[0]
                    st.metric("Average Score", f"{avg_score:.1f}%")
                
                conn.close()
                
            except Exception as e:
                st.error(f"Error loading statistics: {e}")
    
    def _process_resumes(self, uploaded_files, job_id):
        """Process uploaded resumes"""
        try:
            # Get job details
            job = self.db_manager.get_job_by_id(job_id)
            if not job:
                st.error("Job not found")
                return
            
            # Process files
            progress_bar = st.progress(0)
            status_text = st.empty()
            results = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Extract text based on file type
                    text = self._extract_file_text(uploaded_file)
                    
                    if not text or len(text.strip()) < 50:
                        st.warning(f"File {uploaded_file.name} appears to be empty or too short")
                        continue
                    
                    # Parse resume
                    resume = self.resume_parser.parse_resume(text, uploaded_file.name)
                    self.db_manager.save_resume(resume)
                    
                    # Evaluate resume
                    evaluation = self.evaluator.evaluate_resume(resume, job)
                    self.db_manager.save_evaluation(evaluation)
                    
                    results.append({
                        'filename': uploaded_file.name,
                        'candidate': resume.candidate_name,
                        'email': resume.email,
                        'score': evaluation.relevance_score,
                        'verdict': evaluation.verdict
                    })
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    logger.error(f"Error processing {uploaded_file.name}: {e}")
                    continue
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("Processing completed!")
            
            if results:
                st.success(f"Successfully processed {len(results)} resumes")
                
                # Display results
                results_df = pd.DataFrame(results)
                results_df = results_df.sort_values('score', ascending=False)
                st.dataframe(results_df, use_container_width=True)
                
                # Quick statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_fit = len(results_df[results_df['verdict'] == 'High'])
                    st.metric("High Fit", high_fit)
                with col2:
                    avg_score = results_df['score'].mean()
                    st.metric("Average Score", f"{avg_score:.1f}%")
                with col3:
                    top_score = results_df['score'].max()
                    st.metric("Best Score", f"{top_score:.1f}%")
            else:
                st.warning("No resumes were successfully processed")
                
        except Exception as e:
            st.error(f"Processing error: {e}")
            logger.error(f"Processing error: {e}")
    
    def _extract_file_text(self, uploaded_file):
        """Extract text from uploaded file"""
        if uploaded_file.type == "application/pdf":
            return self.doc_processor.extract_text_from_pdf(uploaded_file.getvalue())
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self.doc_processor.extract_text_from_docx(uploaded_file.getvalue())
        elif uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode('utf-8')
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            return ""
    
    def _results_page(self):
        """Results viewing page"""
        st.header("📊 Evaluation Results")
        
        # Get available jobs
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            jobs_df = pd.read_sql_query(
                "SELECT job_id, title, location FROM job_requirements ORDER BY created_at DESC",
                conn
            )
            
            if jobs_df.empty:
                st.warning("No job descriptions found")
                conn.close()
                return
            
            # Job selection and filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                job_options = [f"{row['title']} - {row['location']}" for _, row in jobs_df.iterrows()]
                selected_job_idx = st.selectbox("Select Job", range(len(job_options)), 
                                               format_func=lambda x: job_options[x])
                selected_job_id = jobs_df.iloc[selected_job_idx]['job_id']
            
            with col2:
                min_score = st.slider("Minimum Score", 0, 100, 0)
            
            with col3:
                verdict_filter = st.selectbox("Verdict Filter", ["All", "High", "Medium", "Low"])
            
            # Get evaluation results
            evaluations_df = self._get_filtered_evaluations(selected_job_id, min_score, verdict_filter)
            conn.close()
            
            if evaluations_df.empty:
                st.info("No evaluations found matching your criteria")
                return
            
            st.write(f"Found {len(evaluations_df)} candidates")
            
            # Display results
            for _, eval_row in evaluations_df.head(15).iterrows():
                with st.expander(
                    f"{eval_row['candidate_name']} - Score: {eval_row['relevance_score']:.1f}% ({eval_row['verdict']})",
                    expanded=False
                ):
                    self._render_evaluation_details(eval_row)
            
            # Export functionality
            if st.button("Download Results as CSV"):
                csv_data = evaluations_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"evaluations_{selected_job_id[:8]}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error loading results: {e}")
            logger.error(f"Error loading results: {e}")
    
    def _get_filtered_evaluations(self, job_id, min_score, verdict_filter):
        """Get filtered evaluations from database"""
        conn = sqlite3.connect(self.db_manager.db_path)
        
        query = """
            SELECT e.*, r.candidate_name, r.email, j.title as job_title
            FROM evaluations e
            JOIN resumes r ON e.resume_id = r.resume_id
            JOIN job_requirements j ON e.job_id = j.job_id
            WHERE e.job_id = ?
        """
        params = [job_id]
        
        if min_score > 0:
            query += " AND e.relevance_score >= ?"
            params.append(min_score)
        
        if verdict_filter != "All":
            query += " AND e.verdict = ?"
            params.append(verdict_filter)
        
        query += " ORDER BY e.relevance_score DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def _render_evaluation_details(self, eval_row):
        """Render evaluation details"""
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write(f"**Email:** {eval_row.get('email', 'N/A')}")
            st.write(f"**Relevance Score:** {eval_row['relevance_score']:.1f}/100")
            st.write(f"**Hard Match:** {eval_row.get('hard_match_score', 0):.1f}/100")
            st.write(f"**Semantic Match:** {eval_row.get('semantic_match_score', 0):.1f}/100")
            st.write(f"**Verdict:** {eval_row['verdict']}")
        
        with col_right:
            # Parse JSON fields safely
            try:
                missing_skills = json.loads(eval_row.get('missing_skills', '[]'))
                suggestions = json.loads(eval_row.get('suggestions', '[]'))
                
                if missing_skills:
                    st.write("**Missing Skills:**")
                    for skill in missing_skills[:5]:
                        st.write(f"• {skill}")
                
                if suggestions:
                    st.write("**Suggestions:**")
                    for suggestion in suggestions[:3]:
                        st.write(f"• {suggestion}")
            except json.JSONDecodeError:
                st.write("**Missing Skills:** Data parsing error")
    
    def _system_info_page(self):
        """System information and diagnostics"""
        st.header("⚙️ System Information")
        
        # System component status
        st.subheader("Component Status")
        
        components = check_system_dependencies()
        
        for component, status in components.items():
            if "Available" in status and "Not available" not in status:
                st.success(f"✅ {component}: {status}")
            else:
                st.warning(f"⚠️ {component}: {status}")
        
        # Database statistics
        st.subheader("Database Statistics")
        try:
            stats = self.db_manager.get_database_stats()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Jobs", stats['jobs'])
            with col2:
                st.metric("Total Resumes", stats['resumes'])
            with col3:
                st.metric("Total Evaluations", stats['evaluations'])
            
            # Performance metrics
            if stats['evaluations'] > 0:
                st.subheader("Performance Metrics")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average Relevance Score", f"{stats['avg_score']:.1f}%")
                with col2:
                    st.metric("High Fit Candidates", stats['high_fit'])
            
        except Exception as e:
            st.error(f"Error loading database statistics: {e}")
        
        # Installation help
        st.subheader("Installation Help")
        
        st.write("**To install missing components:**")
        st.code("""
# Basic requirements
pip install streamlit pandas numpy scikit-learn

# Optional but recommended
pip install fuzzywuzzy python-levenshtein  # Better text matching
pip install sentence-transformers          # AI embeddings
pip install PyPDF2                        # PDF processing
pip install docx2txt                      # DOCX processing
pip install plotly                        # Visualizations

# NLP model (run after installing spacy)
python -m spacy download en_core_web_sm
        """)
        
        # Test functionality
        st.subheader("Test System")
        
        if st.button("Run System Test"):
            self._run_system_test()
    
    def _run_system_test(self):
        """Run comprehensive system test"""
        st.write("Running system tests...")
        
        # Test resume parsing
        test_resume = """
        John Doe
        john.doe@email.com
        +1-234-567-8900
        
        EXPERIENCE
        Software Engineer (2 years)
        - Python programming
        - Machine learning projects
        
        SKILLS
        Python, SQL, Machine Learning
        
        EDUCATION
        Bachelor of Computer Science
        """
        
        try:
            resume = self.resume_parser.parse_resume(test_resume, "test.txt")
            st.success("✅ Resume parsing test passed")
            st.write(f"Extracted: {resume.candidate_name}, {len(resume.skills)} skills, {resume.experience_years} years exp")
        except Exception as e:
            st.error(f"❌ Resume parsing test failed: {e}")
        
        # Test job parsing
        test_job = """
        We are looking for a Python Developer with 2+ years of experience.
        Required skills: Python, SQL, Machine Learning
        Preferred: AWS, Docker
        Education: Bachelor's degree in Computer Science
        """
        
        try:
            job = self.jd_parser.parse_job_description(test_job, "Python Developer", "Test Location")
            st.success("✅ Job parsing test passed")
            st.write(f"Extracted: {len(job.must_have_skills)} required skills, {job.experience_years} years exp")
        except Exception as e:
            st.error(f"❌ Job parsing test failed: {e}")
        
        # Test evaluation
        try:
            evaluation = self.evaluator.evaluate_resume(resume, job)
            st.success("✅ Evaluation test passed")
            st.write(f"Score: {evaluation.relevance_score:.1f}%, Verdict: {evaluation.verdict}")
        except Exception as e:
            st.error(f"❌ Evaluation test failed: {e}")

"""
AI-Powered Resume Relevance Check System
Main application file for Innomatics Research Labs
"""

import streamlit as st
import traceback
from src.app import ResumeRelevanceApp
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

def main():
    """Main function with comprehensive error handling"""
    try:
        # Set up the page first
        st.set_page_config(
            page_title="Resume Relevance System",
            page_icon="📄",
            layout="wide"
        )
        
        # Show loading message
        loading_placeholder = st.empty()
        loading_placeholder.info("Initializing application...")
        
        # Initialize app
        app = ResumeRelevanceApp()
        
        # Clear loading message and run app
        loading_placeholder.empty()
        app.run()
        
    except Exception as e:
        st.error("Application startup failed!")
        st.error(f"Error details: {str(e)}")
        logger.error(f"Application startup failed: {e}")
        
        # Show helpful information
        st.markdown("### Troubleshooting Steps:")
        st.markdown("""
        1. **Check Python version**: Ensure you're using Python 3.8+
        2. **Install required packages**:
           ```bash
           pip install -r requirements.txt
           ```
        3. **Install optional packages** for full functionality:
           ```bash
           pip install fuzzywuzzy python-levenshtein sentence-transformers
           pip install PyPDF2 docx2txt plotly
           python -m spacy download en_core_web_sm
           ```
        4. **Check file permissions**: Ensure the app can create database files
        5. **Try running with**: `streamlit run main.py --server.enableCORS=false`
        """)
        
        # Show system information
        st.markdown("### System Information:")
        try:
            import pandas as pd
            st.write(f"Pandas version: {pd.__version__}")
            st.write(f"Streamlit version: {st.__version__}")
        except ImportError:
            st.write("Core packages not properly installed")
        
        # Show detailed error for debugging
        with st.expander("Detailed Error Information"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

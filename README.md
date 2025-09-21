Automated Resume Relevance Check System
📌 Problem Statement

At Innomatics Research Labs, resume evaluation is currently manual, inconsistent, and time-consuming. Each week, the placement team across Hyderabad, Bangalore, Pune, and Delhi NCR receives 18–20 job requirements, with each posting attracting hundreds of applications.

Currently, recruiters manually review resumes against job descriptions (JD), leading to:

Delays in shortlisting candidates

Human bias and inconsistency

Missed qualified applicants

Additional workload for mentors and placement teams

🚀 Proposed Solution

We propose an AI-driven Resume Relevance Check System that:

Accepts student resumes (PDF/DOCX) and job descriptions uploaded by placement teams

Extracts text and converts it into embeddings

Performs a hybrid evaluation:

Hard Match: Skills, education, and keyword overlap

Soft Match: Semantic fit using embeddings + LLM reasoning

Generates:

✅ Relevance Score (0–100%)

⚠️ Missing Skills / Qualifications

📝 Suggestions for improvement

🔎 Verdict (High / Medium / Low fit)

Stores results in a dashboard searchable by recruiters

This ensures fair, quick, and consistent evaluation for all candidates.

🛠️ Workflow

Upload resume + job description

Extract & Preprocess text from PDF/DOCX

Vectorize & Compare using TF-IDF + Embeddings

Hybrid Scoring (keywords + semantic match)

Generate Output: Relevance score, missing skills, suggestions

Store Results in database for recruiters

🎨 User Interface (UI)

Streamlit-based Web App

File upload section for resumes & JDs

Dashboard view for recruiters

Graphs & insights on candidate-job match

Searchable results history

🏗️ Tech Stack

Frontend: Streamlit

Backend: Python (Flask/FastAPI optional)

NLP & AI:

scikit-learn (TF-IDF, cosine similarity)

OpenAI/LLM embeddings for semantic analysis

Database: SQLite (for demo) / PostgreSQL (production)

Other Libraries: PyPDF2, docx2txt, pandas, numpy

⚙️ Installation Steps

Clone the repository:

git clone https://github.com/your-username/resume-relevance-check.git
cd resume-relevance-check


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows


Install dependencies:

pip install -r requirements.txt


Add your OpenAI API key (if embeddings are used):

export OPENAI_API_KEY="your_api_key_here"   # Mac/Linux
setx OPENAI_API_KEY "your_api_key_here"     # Windows

▶️ Usage

Run the Streamlit app:

streamlit run app.py


Upload a resume and a job description

View the Relevance Score, Missing Skills, and Verdict

Check the dashboard for all past evaluations

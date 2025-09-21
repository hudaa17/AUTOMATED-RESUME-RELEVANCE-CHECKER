import os
import json
import sqlite3
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import logging

# Core libraries
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from fuzzywuzzy import fuzz
import streamlit as st

# Document processing
import PyPDF2
import docx2txt
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model (download with: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

@dataclass
class JobRequirement:
    """Data structure for job requirements"""
    job_id: str
    title: str
    must_have_skills: List[str]
    good_to_have_skills: List[str]
    qualifications: List[str]
    experience_years: int
    description: str
    location: str

@dataclass
class ResumeData:
    """Data structure for resume information"""
    resume_id: str
    candidate_name: str
    email: str
    phone: str
    skills: List[str]
    experience_years: float
    education: List[str]
    projects: List[str]
    certifications: List[str]
    raw_text: str

@dataclass
class EvaluationResult:
    """Data structure for evaluation results"""
    resume_id: str
    job_id: str
    relevance_score: float
    verdict: str  # High/Medium/Low
    missing_skills: List[str]
    missing_qualifications: List[str]
    suggestions: List[str]
    hard_match_score: float
    semantic_match_score: float
    timestamp: str

class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str = "resume_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Job requirements table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS job_requirements (
                job_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                must_have_skills TEXT,
                good_to_have_skills TEXT,
                qualifications TEXT,
                experience_years INTEGER,
                description TEXT,
                location TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Resume data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resumes (
                resume_id TEXT PRIMARY KEY,
                candidate_name TEXT,
                email TEXT,
                phone TEXT,
                skills TEXT,
                experience_years REAL,
                education TEXT,
                projects TEXT,
                certifications TEXT,
                raw_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Evaluation results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                resume_id TEXT,
                job_id TEXT,
                relevance_score REAL,
                verdict TEXT,
                missing_skills TEXT,
                missing_qualifications TEXT,
                suggestions TEXT,
                hard_match_score REAL,
                semantic_match_score REAL,
                timestamp TEXT,
                FOREIGN KEY (resume_id) REFERENCES resumes (resume_id),
                FOREIGN KEY (job_id) REFERENCES job_requirements (job_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_job_requirement(self, job: JobRequirement):
        """Save job requirement to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO job_requirements 
            (job_id, title, must_have_skills, good_to_have_skills, qualifications, 
             experience_years, description, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job.job_id, job.title,
            json.dumps(job.must_have_skills),
            json.dumps(job.good_to_have_skills),
            json.dumps(job.qualifications),
            job.experience_years,
            job.description,
            job.location
        ))
        
        conn.commit()
        conn.close()
    
    def save_resume(self, resume: ResumeData):
        """Save resume data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO resumes 
            (resume_id, candidate_name, email, phone, skills, experience_years, 
             education, projects, certifications, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            resume.resume_id, resume.candidate_name, resume.email, resume.phone,
            json.dumps(resume.skills), resume.experience_years,
            json.dumps(resume.education), json.dumps(resume.projects),
            json.dumps(resume.certifications), resume.raw_text
        ))
        
        conn.commit()
        conn.close()
    
    def save_evaluation(self, evaluation: EvaluationResult):
        """Save evaluation result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evaluations 
            (resume_id, job_id, relevance_score, verdict, missing_skills, 
             missing_qualifications, suggestions, hard_match_score, 
             semantic_match_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evaluation.resume_id, evaluation.job_id, evaluation.relevance_score,
            evaluation.verdict, json.dumps(evaluation.missing_skills),
            json.dumps(evaluation.missing_qualifications),
            json.dumps(evaluation.suggestions), evaluation.hard_match_score,
            evaluation.semantic_match_score, evaluation.timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_evaluations_by_job(self, job_id: str) -> List[Dict]:
        """Get all evaluations for a specific job"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT e.*, r.candidate_name, r.email, j.title
            FROM evaluations e
            JOIN resumes r ON e.resume_id = r.resume_id
            JOIN job_requirements j ON e.job_id = j.job_id
            WHERE e.job_id = ?
            ORDER BY e.relevance_score DESC
        ''', (job_id,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

class DocumentProcessor:
    """Handles document parsing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(file_content: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = BytesIO(file_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            return ""
    
    @staticmethod
    def extract_text_from_docx(file_content: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            docx_file = BytesIO(file_content)
            text = docx2txt.process(docx_file)
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX: {e}")
            return ""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.,;:()\-@+]', '', text)
        return text.strip()

class ResumeParser:
    """Parse resume content and extract structured information"""
    
    def __init__(self):
        self.skills_keywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'node.js', 'sql',
            'machine learning', 'deep learning', 'data analysis', 'pandas', 'numpy',
            'tensorflow', 'pytorch', 'scikit-learn', 'aws', 'azure', 'docker',
            'kubernetes', 'git', 'html', 'css', 'mongodb', 'postgresql', 'mysql'
        ]
    
    def parse_resume(self, text: str, filename: str) -> ResumeData:
        """Parse resume text and extract structured information"""
        resume_id = hashlib.md5(f"{filename}{text[:100]}".encode()).hexdigest()
        
        # Extract basic information
        candidate_name = self._extract_name(text)
        email = self._extract_email(text)
        phone = self._extract_phone(text)
        
        # Extract structured information
        skills = self._extract_skills(text)
        experience_years = self._extract_experience_years(text)
        education = self._extract_education(text)
        projects = self._extract_projects(text)
        certifications = self._extract_certifications(text)
        
        return ResumeData(
            resume_id=resume_id,
            candidate_name=candidate_name,
            email=email,
            phone=phone,
            skills=skills,
            experience_years=experience_years,
            education=education,
            projects=projects,
            certifications=certifications,
            raw_text=text
        )
    
    def _extract_name(self, text: str) -> str:
        """Extract candidate name from resume"""
        lines = text.split('\n')
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if len(line) > 0 and len(line.split()) <= 4:
                # Simple heuristic: name is likely in first few lines
                if not re.search(r'@|phone|email|address', line.lower()):
                    return line
        return "Unknown"
    
    def _extract_email(self, text: str) -> str:
        """Extract email from resume"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, text)
        return matches[0] if matches else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from resume"""
        phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        matches = re.findall(phone_pattern, text)
        return matches[0] if matches else ""
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skills_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        # Also look for skills in dedicated skills section
        skills_section = self._find_section(text, ['skills', 'technical skills', 'technologies'])
        if skills_section:
            # Extract comma-separated skills
            skill_matches = re.findall(r'\b[A-Za-z][A-Za-z\s\.\-+]{1,20}\b', skills_section)
            found_skills.extend([s.strip() for s in skill_matches if len(s.strip()) > 2])
        
        return list(set(found_skills))  # Remove duplicates
    
    def _extract_experience_years(self, text: str) -> float:
        """Extract years of experience"""
        # Look for patterns like "5 years experience", "3+ years", etc.
        patterns = [
            r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience[:\s]*(\d+(?:\.\d+)?)\+?\s*years?',
            r'(\d+(?:\.\d+)?)\+?\s*years?\s+in'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return float(matches[0])
        
        return 0.0
    
    def _extract_education(self, text: str) -> List[str]:
        """Extract education information"""
        education_section = self._find_section(text, ['education', 'academic', 'qualification'])
        if not education_section:
            return []
        
        # Look for degree patterns
        degree_patterns = [
            r'\b(?:bachelor|master|phd|doctorate|b\.?tech|m\.?tech|mba|bca|mca)\b.*',
            r'\b(?:b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)\b.*'
        ]
        
        education = []
        for pattern in degree_patterns:
            matches = re.findall(pattern, education_section, re.IGNORECASE)
            education.extend(matches)
        
        return education
    
    def _extract_projects(self, text: str) -> List[str]:
        """Extract project information"""
        projects_section = self._find_section(text, ['projects', 'project work', 'academic projects'])
        if not projects_section:
            return []
        
        # Split by bullet points or line breaks
        project_lines = re.split(r'[•\-\n]', projects_section)
        projects = [line.strip() for line in project_lines if len(line.strip()) > 10]
        
        return projects[:5]  # Limit to 5 projects
    
    def _extract_certifications(self, text: str) -> List[str]:
        """Extract certification information"""
        cert_section = self._find_section(text, ['certification', 'certificate', 'license'])
        if not cert_section:
            return []
        
        # Split by bullet points or line breaks
        cert_lines = re.split(r'[•\-\n]', cert_section)
        certifications = [line.strip() for line in cert_lines if len(line.strip()) > 5]
        
        return certifications
    
    def _find_section(self, text: str, section_names: List[str]) -> str:
        """Find a specific section in the resume"""
        text_lines = text.split('\n')
        section_content = ""
        in_section = False
        
        for i, line in enumerate(text_lines):
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            if any(name in line_lower for name in section_names):
                in_section = True
                continue
            
            # Check if we've hit another section
            if in_section and line_lower.endswith(':') and len(line.strip()) < 30:
                break
            
            if in_section:
                section_content += line + "\n"
        
        return section_content.strip()

class JobDescriptionParser:
    """Parse job description and extract requirements"""
    
    def parse_job_description(self, text: str, job_title: str, location: str = "") -> JobRequirement:
        """Parse job description and extract structured requirements"""
        job_id = hashlib.md5(f"{job_title}{text[:100]}".encode()).hexdigest()
        
        must_have_skills = self._extract_must_have_skills(text)
        good_to_have_skills = self._extract_good_to_have_skills(text)
        qualifications = self._extract_qualifications(text)
        experience_years = self._extract_required_experience(text)
        
        return JobRequirement(
            job_id=job_id,
            title=job_title,
            must_have_skills=must_have_skills,
            good_to_have_skills=good_to_have_skills,
            qualifications=qualifications,
            experience_years=experience_years,
            description=text,
            location=location
        )
    
    def _extract_must_have_skills(self, text: str) -> List[str]:
        """Extract must-have skills from job description"""
        # Look for sections with "required", "must have", "essential"
        required_section = self._find_requirements_section(text, ['required', 'must have', 'essential'])
        skills = self._extract_skills_from_text(required_section or text)
        return skills[:10]  # Limit to 10 skills
    
    def _extract_good_to_have_skills(self, text: str) -> List[str]:
        """Extract good-to-have skills from job description"""
        # Look for sections with "preferred", "nice to have", "plus"
        preferred_section = self._find_requirements_section(text, ['preferred', 'nice to have', 'plus', 'bonus'])
        skills = self._extract_skills_from_text(preferred_section or "")
        return skills[:8]  # Limit to 8 skills
    
    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract qualification requirements"""
        qual_section = self._find_requirements_section(text, ['qualification', 'education', 'degree'])
        
        qualifications = []
        if qual_section:
            # Look for degree patterns
            degree_patterns = [
                r'\b(?:bachelor|master|phd|b\.?tech|m\.?tech|mba|bca|mca).*',
                r'\b(?:computer science|engineering|mathematics|statistics).*'
            ]
            
            for pattern in degree_patterns:
                matches = re.findall(pattern, qual_section, re.IGNORECASE)
                qualifications.extend(matches)
        
        return qualifications
    
    def _extract_required_experience(self, text: str) -> int:
        """Extract required years of experience"""
        patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'minimum\s+(\d+)\s*years?',
            r'at least\s+(\d+)\s*years?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return int(matches[0])
        
        return 0
    
    def _find_requirements_section(self, text: str, keywords: List[str]) -> Optional[str]:
        """Find requirements section in job description"""
        text_lines = text.split('\n')
        section_content = ""
        in_section = False
        
        for line in text_lines:
            line_lower = line.lower().strip()
            
            if any(keyword in line_lower for keyword in keywords):
                in_section = True
                continue
            
            if in_section:
                if line.strip() == "" or (line_lower.endswith(':') and len(line.strip()) < 40):
                    if len(section_content.strip()) > 50:  # Got enough content
                        break
                else:
                    section_content += line + "\n"
        
        return section_content.strip() if section_content.strip() else None
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text using patterns"""
        if not text:
            return []
        
        # Common technical skills
        tech_skills = [
            'python', 'java', 'javascript', 'react', 'angular', 'node.js', 'sql',
            'machine learning', 'deep learning', 'data analysis', 'pandas', 'numpy',
            'tensorflow', 'pytorch', 'aws', 'azure', 'docker', 'kubernetes', 'git'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        for skill in tech_skills:
            if skill in text_lower:
                found_skills.append(skill)
        
        # Extract other skills using patterns
        skill_patterns = [
            r'\b[A-Z][A-Za-z\s\.\-+]{2,20}\b',  # Capitalized terms
            r'\b\w+\.js\b',  # JavaScript frameworks
            r'\b\w+SQL\b',   # SQL variants
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, text)
            found_skills.extend([match.strip() for match in matches if len(match.strip()) > 2])
        
        return list(set(found_skills))

class RelevanceEvaluator:
    """Evaluate resume relevance against job requirements"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def evaluate_resume(self, resume: ResumeData, job: JobRequirement) -> EvaluationResult:
        """Evaluate resume against job requirements"""
        
        # Calculate hard match score (keyword/skill matching)
        hard_score = self._calculate_hard_match_score(resume, job)
        
        # Calculate semantic match score (text similarity)
        semantic_score = self._calculate_semantic_match_score(resume, job)
        
        # Combine scores with weights
        relevance_score = (hard_score * 0.6) + (semantic_score * 0.4)
        
        # Determine verdict
        verdict = self._determine_verdict(relevance_score)
        
        # Find missing elements
        missing_skills = self._find_missing_skills(resume, job)
        missing_qualifications = self._find_missing_qualifications(resume, job)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(resume, job, missing_skills, missing_qualifications)
        
        return EvaluationResult(
            resume_id=resume.resume_id,
            job_id=job.job_id,
            relevance_score=relevance_score,
            verdict=verdict,
            missing_skills=missing_skills,
            missing_qualifications=missing_qualifications,
            suggestions=suggestions,
            hard_match_score=hard_score,
            semantic_match_score=semantic_score,
            timestamp=datetime.now().isoformat()
        )
    
    def _calculate_hard_match_score(self, resume: ResumeData, job: JobRequirement) -> float:
        """Calculate hard match score based on exact and fuzzy keyword matching"""
        score = 0.0
        total_possible = 0
        
        # Skills matching
        must_have_matches = 0
        for required_skill in job.must_have_skills:
            total_possible += 2  # Must-have skills are weighted more
            for resume_skill in resume.skills:
                if fuzz.ratio(required_skill.lower(), resume_skill.lower()) > 80:
                    must_have_matches += 2
                    break
        
        good_to_have_matches = 0
        for preferred_skill in job.good_to_have_skills:
            total_possible += 1  # Good-to-have skills have lower weight
            for resume_skill in resume.skills:
                if fuzz.ratio(preferred_skill.lower(), resume_skill.lower()) > 80:
                    good_to_have_matches += 1
                    break
        
        # Experience matching
        total_possible += 2
        if resume.experience_years >= job.experience_years:
            score += 2
        elif resume.experience_years >= job.experience_years * 0.8:
            score += 1.5
        elif resume.experience_years >= job.experience_years * 0.6:
            score += 1
        
        # Education matching
        total_possible += 1
        if job.qualifications:
            for qual in job.qualifications:
                for edu in resume.education:
                    if fuzz.partial_ratio(qual.lower(), edu.lower()) > 70:
                        score += 1
                        break
                break
        
        score += must_have_matches + good_to_have_matches
        
        return min(100, (score / max(total_possible, 1)) * 100) if total_possible > 0 else 0
    
    def _calculate_semantic_match_score(self, resume: ResumeData, job: JobRequirement) -> float:
        """Calculate semantic match score using TF-IDF and cosine similarity"""
        try:
            documents = [resume.raw_text, job.description]
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0]) * 100
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
            return 50.0  # Default score
    
    def _determine_verdict(self, score: float) -> str:
        """Determine verdict based on relevance score"""
        if score >= 75:
            return "High"
        elif score >= 50:
            return "Medium"
        else:
            return "Low"
    
    def _find_missing_skills(self, resume: ResumeData, job: JobRequirement) -> List[str]:
        """Find missing skills from job requirements"""
        missing = []
        resume_skills_lower = [s.lower() for s in resume.skills]
        
        for required_skill in job.must_have_skills:
            if not any(fuzz.ratio(required_skill.lower(), resume_skill) > 80 
                      for resume_skill in resume_skills_lower):
                missing.append(required_skill)
        
        return missing
    
    def _find_missing_qualifications(self, resume: ResumeData, job: JobRequirement) -> List[str]:
        """Find missing qualifications from job requirements"""
        missing = []
        
        for qual in job.qualifications:
            if not any(fuzz.partial_ratio(qual.lower(), edu.lower()) > 70 
                      for edu in resume.education):
                missing.append(qual)
        
        return missing
    
    def _generate_suggestions(self, resume: ResumeData, job: JobRequirement, 
                            missing_skills: List[str], missing_qualifications: List[str]) -> List[str]:
        """Generate improvement suggestions for the candidate"""
        suggestions = []
        
        if missing_skills:
            suggestions.append(f"Consider learning these key skills: {', '.join(missing_skills[:3])}")
        
        if resume.experience_years < job.experience_years:
            gap = job.experience_years - resume.experience_years
            suggestions.append(f"Gain {gap:.1f} more years of relevant experience")
        
        if missing_qualifications:
            suggestions.append(f"Consider pursuing: {', '.join(missing_qualifications[:2])}")
        
        if len(resume.projects) < 3:
            suggestions.append("Add more relevant projects to showcase your skills")
        
        if len(resume.certifications) == 0:
            suggestions.append("Consider getting industry-relevant certifications")
        
        return suggestions

# Streamlit Web Application
class ResumeRelevanceApp:
    """Main Streamlit application"""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.doc_processor = DocumentProcessor()
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()
        self.evaluator = RelevanceEvaluator()
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Resume Relevance Check System",
            page_icon="📄",
            layout="wide"
        )
        
        st.title("🎯 Automated Resume Relevance Check System")
        st.markdown("**Innomatics Research Labs** - Placement Team Dashboard")
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Page",
            ["Upload Job Description", "Upload Resumes", "View Results", "Analytics"]
        )
        
        if page == "Upload Job Description":
            self._job_upload_page()
        elif page == "Upload Resumes":
            self._resume_upload_page()
        elif page == "View Results":
            self._results_page()
        elif page == "Analytics":
            self._analytics_page()
    
    def _job_upload_page(self):
        """Job description upload and management page"""
        st.header("📋 Job Description Management")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload New Job Requirement")
            
            job_title = st.text_input("Job Title", placeholder="e.g., Data Scientist")
            location = st.selectbox(
                "Location",
                ["Hyderabad", "Bangalore", "Pune", "Delhi NCR"]
            )
            
            jd_text = st.text_area(
                "Job Description",
                height=300,
                placeholder="Paste the complete job description here..."
            )
            
            if st.button("Parse and Save Job Description", type="primary"):
                if job_title and jd_text:
                    try:
                        job = self.jd_parser.parse_job_description(jd_text, job_title, location)
                        self.db_manager.save_job_requirement(job)
                        
                        st.success("✅ Job description parsed and saved successfully!")
                        
                        # Display parsed information
                        st.subheader("Parsed Information")
                        st.write(f"**Job ID:** {job.job_id}")
                        st.write(f"**Must-have Skills:** {', '.join(job.must_have_skills)}")
                        st.write(f"**Good-to-have Skills:** {', '.join(job.good_to_have_skills)}")
                        st.write(f"**Required Experience:** {job.experience_years} years")
                        st.write(f"**Qualifications:** {', '.join(job.qualifications)}")
                        
                    except Exception as e:
                        st.error(f"❌ Error parsing job description: {e}")
                else:
                    st.warning("⚠️ Please fill in all required fields")
        
        with col2:
            st.subheader("Recent Job Postings")
            # Display recent job postings from database
            try:
                conn = sqlite3.connect(self.db_manager.db_path)
                df = pd.read_sql_query(
                    "SELECT job_id, title, location, created_at FROM job_requirements ORDER BY created_at DESC LIMIT 10",
                    conn
                )
                conn.close()
                
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No job postings found. Upload your first job description!")
                    
            except Exception as e:
                st.error(f"Error loading job postings: {e}")
    
    def _resume_upload_page(self):
        """Resume upload and processing page"""
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
                st.warning("⚠️ No job descriptions found. Please upload a job description first.")
                return
                
        except Exception as e:
            st.error(f"Error loading jobs: {e}")
            return
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Upload Resumes")
            
            # Job selection
            job_options = [f"{row['title']} - {row['location']} ({row['job_id'][:8]}...)" 
                          for _, row in jobs_df.iterrows()]
            selected_job_idx = st.selectbox("Select Job Posting", range(len(job_options)), 
                                           format_func=lambda x: job_options[x])
            selected_job_id = jobs_df.iloc[selected_job_idx]['job_id']
            
            # File uploader
            uploaded_files = st.file_uploader(
                "Upload Resume Files",
                type=['pdf', 'docx'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.info(f"📁 {len(uploaded_files)} files uploaded")
                
                if st.button("Process Resumes", type="primary"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Get job details
                    conn = sqlite3.connect(self.db_manager.db_path)
                    job_data = conn.execute(
                        "SELECT * FROM job_requirements WHERE job_id = ?", 
                        (selected_job_id,)
                    ).fetchone()
                    conn.close()
                    
                    if job_data:
                        job = JobRequirement(
                            job_id=job_data[0],
                            title=job_data[1],
                            must_have_skills=json.loads(job_data[2] or '[]'),
                            good_to_have_skills=json.loads(job_data[3] or '[]'),
                            qualifications=json.loads(job_data[4] or '[]'),
                            experience_years=job_data[5] or 0,
                            description=job_data[6] or '',
                            location=job_data[7] or ''
                        )
                        
                        processed_count = 0
                        results = []
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            try:
                                status_text.text(f"Processing {uploaded_file.name}...")
                                
                                # Extract text based on file type
                                if uploaded_file.type == "application/pdf":
                                    text = self.doc_processor.extract_text_from_pdf(uploaded_file.getvalue())
                                elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                                    text = self.doc_processor.extract_text_from_docx(uploaded_file.getvalue())
                                else:
                                    continue
                                
                                if text.strip():
                                    # Clean text
                                    cleaned_text = self.doc_processor.clean_text(text)
                                    
                                    # Parse resume
                                    resume = self.resume_parser.parse_resume(cleaned_text, uploaded_file.name)
                                    
                                    # Save resume to database
                                    self.db_manager.save_resume(resume)
                                    
                                    # Evaluate resume
                                    evaluation = self.evaluator.evaluate_resume(resume, job)
                                    
                                    # Save evaluation
                                    self.db_manager.save_evaluation(evaluation)
                                    
                                    results.append({
                                        'filename': uploaded_file.name,
                                        'candidate': resume.candidate_name,
                                        'score': evaluation.relevance_score,
                                        'verdict': evaluation.verdict
                                    })
                                    
                                    processed_count += 1
                                
                                # Update progress
                                progress_bar.progress((i + 1) / len(uploaded_files))
                                
                            except Exception as e:
                                st.error(f"Error processing {uploaded_file.name}: {e}")
                        
                        status_text.text("✅ Processing completed!")
                        st.success(f"Processed {processed_count} resumes successfully!")
                        
                        # Display results summary
                        if results:
                            st.subheader("Processing Results")
                            results_df = pd.DataFrame(results)
                            st.dataframe(results_df.sort_values('score', ascending=False), use_container_width=True)
        
        with col2:
            st.subheader("Processing Statistics")
            
            # Show statistics about processed resumes
            try:
                conn = sqlite3.connect(self.db_manager.db_path)
                
                # Total resumes processed
                total_resumes = conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
                st.metric("Total Resumes", total_resumes)
                
                # Total evaluations
                total_evaluations = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
                st.metric("Total Evaluations", total_evaluations)
                
                # Recent evaluations
                recent_evals = pd.read_sql_query('''
                    SELECT r.candidate_name, j.title, e.relevance_score, e.verdict
                    FROM evaluations e
                    JOIN resumes r ON e.resume_id = r.resume_id
                    JOIN job_requirements j ON e.job_id = j.job_id
                    ORDER BY e.timestamp DESC
                    LIMIT 10
                ''', conn)
                
                conn.close()
                
                if not recent_evals.empty:
                    st.subheader("Recent Evaluations")
                    st.dataframe(recent_evals, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading statistics: {e}")
    
    def _results_page(self):
        """Results viewing and filtering page"""
        st.header("📊 Evaluation Results")
        
        # Get available jobs
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            jobs_df = pd.read_sql_query(
                "SELECT job_id, title, location FROM job_requirements ORDER BY created_at DESC",
                conn
            )
            conn.close()
            
            if jobs_df.empty:
                st.warning("⚠️ No job descriptions found.")
                return
                
        except Exception as e:
            st.error(f"Error loading jobs: {e}")
            return
        
        # Job selection
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            job_options = [f"{row['title']} - {row['location']}" for _, row in jobs_df.iterrows()]
            selected_job_idx = st.selectbox("Select Job for Results", range(len(job_options)), 
                                           format_func=lambda x: job_options[x])
            selected_job_id = jobs_df.iloc[selected_job_idx]['job_id']
        
        with col2:
            min_score = st.slider("Minimum Score", 0, 100, 0)
        
        with col3:
            verdict_filter = st.selectbox("Verdict Filter", ["All", "High", "Medium", "Low"])
        
        # Get evaluation results
        try:
            evaluations = self.db_manager.get_evaluations_by_job(selected_job_id)
            
            if not evaluations:
                st.info("No evaluations found for this job.")
                return
            
            # Filter results
            filtered_evals = evaluations
            if min_score > 0:
                filtered_evals = [e for e in filtered_evals if e['relevance_score'] >= min_score]
            if verdict_filter != "All":
                filtered_evals = [e for e in filtered_evals if e['verdict'] == verdict_filter]
            
            if not filtered_evals:
                st.warning("No results match your filters.")
                return
            
            st.subheader(f"Results: {len(filtered_evals)} candidates found")
            
            # Display results in cards
            for eval_data in filtered_evals[:20]:  # Limit to top 20 results
                with st.expander(
                    f"🎯 {eval_data['candidate_name']} - Score: {eval_data['relevance_score']:.1f} ({eval_data['verdict']})",
                    expanded=False
                ):
                    col_left, col_right = st.columns([1, 1])
                    
                    with col_left:
                        st.write(f"**Email:** {eval_data.get('email', 'N/A')}")
                        st.write(f"**Relevance Score:** {eval_data['relevance_score']:.1f}/100")
                        st.write(f"**Verdict:** {eval_data['verdict']}")
                        st.write(f"**Hard Match Score:** {eval_data.get('hard_match_score', 0):.1f}")
                        st.write(f"**Semantic Match Score:** {eval_data.get('semantic_match_score', 0):.1f}")
                    
                    with col_right:
                        # Parse JSON fields
                        missing_skills = json.loads(eval_data.get('missing_skills', '[]'))
                        suggestions = json.loads(eval_data.get('suggestions', '[]'))
                        
                        if missing_skills:
                            st.write("**Missing Skills:**")
                            for skill in missing_skills[:5]:
                                st.write(f"• {skill}")
                        
                        if suggestions:
                            st.write("**Improvement Suggestions:**")
                            for suggestion in suggestions[:3]:
                                st.write(f"• {suggestion}")
            
            # Download results
            if st.button("📥 Download Results as CSV"):
                df = pd.DataFrame(filtered_evals)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"resume_evaluations_{selected_job_id[:8]}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Error loading results: {e}")
    
    def _analytics_page(self):
        """Analytics and insights page"""
        st.header("📈 Analytics Dashboard")
        
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            
            # Overall statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_jobs = conn.execute("SELECT COUNT(*) FROM job_requirements").fetchone()[0]
                st.metric("Total Jobs", total_jobs)
            
            with col2:
                total_resumes = conn.execute("SELECT COUNT(*) FROM resumes").fetchone()[0]
                st.metric("Total Resumes", total_resumes)
            
            with col3:
                total_evaluations = conn.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0]
                st.metric("Total Evaluations", total_evaluations)
            
            with col4:
                avg_score = conn.execute("SELECT AVG(relevance_score) FROM evaluations").fetchone()[0]
                st.metric("Average Score", f"{avg_score:.1f}" if avg_score else "0.0")
            
            # Score distribution
            st.subheader("Score Distribution")
            scores_df = pd.read_sql_query(
                "SELECT relevance_score, verdict FROM evaluations",
                conn
            )
            
            if not scores_df.empty:
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    st.bar_chart(scores_df['relevance_score'].value_counts().sort_index())
                
                with col_chart2:
                    verdict_counts = scores_df['verdict'].value_counts()
                    st.bar_chart(verdict_counts)
            
            # Top performing candidates
            st.subheader("Top Performing Candidates")
            top_candidates = pd.read_sql_query('''
                SELECT r.candidate_name, r.email, j.title, e.relevance_score, e.verdict
                FROM evaluations e
                JOIN resumes r ON e.resume_id = r.resume_id
                JOIN job_requirements j ON e.job_id = j.job_id
                ORDER BY e.relevance_score DESC
                LIMIT 10
            ''', conn)
            
            if not top_candidates.empty:
                st.dataframe(top_candidates, use_container_width=True)
            
            # Job-wise performance
            st.subheader("Job-wise Performance")
            job_performance = pd.read_sql_query('''
                SELECT j.title, j.location,
                       COUNT(e.id) as total_applications,
                       AVG(e.relevance_score) as avg_score,
                       SUM(CASE WHEN e.verdict = 'High' THEN 1 ELSE 0 END) as high_fit_count
                FROM job_requirements j
                LEFT JOIN evaluations e ON j.job_id = e.job_id
                GROUP BY j.job_id, j.title, j.location
                ORDER BY avg_score DESC
            ''', conn)
            
            if not job_performance.empty:
                st.dataframe(job_performance, use_container_width=True)
            
            conn.close()
            
        except Exception as e:
            st.error(f"Error loading analytics: {e}")

# Main execution
def main():
    """Main function to run the application"""
    app = ResumeRelevanceApp()
    app.run()

if __name__ == "__main__":
    main()






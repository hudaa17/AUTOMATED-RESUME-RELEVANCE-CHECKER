lass DatabaseManager:
    def __init__(self, db_path: str = "resume_system_llm.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS job_requirements (
            job_id TEXT PRIMARY KEY,
            title TEXT, must_have_skills TEXT, good_to_have_skills TEXT,
            qualifications TEXT, experience_years INTEGER, description TEXT,
            location TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS resumes (
            resume_id TEXT PRIMARY KEY,
            candidate_name TEXT, email TEXT, phone TEXT,
            skills TEXT, experience_years REAL, education TEXT,
            projects TEXT, certifications TEXT, raw_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            resume_id TEXT, job_id TEXT, relevance_score REAL,
            verdict TEXT, missing_skills TEXT, missing_qualifications TEXT,
            suggestions TEXT, llm_feedback TEXT, timestamp TEXT,
            FOREIGN KEY(resume_id) REFERENCES resumes(resume_id),
            FOREIGN KEY(job_id) REFERENCES job_requirements(job_id)
        )''')
        # (Removed stray conn.commit() and conn.close() lines)
    
    def save_job_requirement(self, job: JobRequirement):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''INSERT OR REPLACE INTO job_requirements
            (job_id, title, must_have_skills, good_to_have_skills, qualifications,
             experience_years, description, location)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (job.job_id, job.title, json.dumps(job.must_have_skills),
             json.dumps(job.good_to_have_skills), json.dumps(job.qualifications),
             job.experience_years, job.description, job.location))
        # Removed stray conn.commit() and conn.close() lines
    
    def save_resume(self, resume: ResumeData):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''INSERT OR REPLACE INTO resumes
            (resume_id, candidate_name, email, phone, skills, experience_years,
             education, projects, certifications, raw_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (resume.resume_id, resume.candidate_name, resume.email, resume.phone,
             json.dumps(resume.skills), resume.experience_years,
             json.dumps(resume.education), json.dumps(resume.projects),
             json.dumps(resume.certifications), resume.raw_text))
        # Removed stray conn.commit() and conn.close() lines
    
    def save_evaluation(self, evaluation: EvaluationResult):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO evaluations
            (resume_id, job_id, relevance_score, verdict, missing_skills,
             missing_qualifications, suggestions, llm_feedback, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                evaluation.resume_id,
                evaluation.job_id,
                evaluation.relevance_score,
                evaluation.verdict,
                json.dumps(evaluation.missing_skills),           # store list as JSON string
                json.dumps(evaluation.missing_qualifications),  # same here
                json.dumps(evaluation.suggestions),
                evaluation.llm_feedback,
                evaluation.timestamp
            )
        )

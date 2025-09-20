@dataclass
class JobRequirement:
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
    resume_id: str
    job_id: str
    relevance_score: float
    verdict: str
    missing_skills: List[str]
    missing_qualifications: List[str]
    suggestions: List[str]
    llm_feedback: str
    timestamp: str



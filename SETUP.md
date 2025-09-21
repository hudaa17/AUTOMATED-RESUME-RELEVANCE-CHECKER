# Setup Guide for Resume Relevance Check System

## Prerequisites

Before setting up the system, ensure you have:

- **Python 3.8+** installed on your system
- **Git** for version control
- At least **2GB RAM** available
- **500MB disk space** for dependencies and data

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/resume-relevance-system.git
cd resume-relevance-system
```

### 2. Create Virtual Environment

**On Windows:**
```bash
python -m venv resume_env
resume_env\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv resume_env
source resume_env/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Verify Installation

```bash
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ spaCy model loaded successfully')"
```

### 6. Create Environment Configuration (Optional)

Create a `.env` file in the project root:

```bash
# Database settings
DATABASE_PATH=data/resume_system.db

# Application settings
MAX_UPLOAD_SIZE_MB=50
LOG_LEVEL=INFO
DEBUG_MODE=False

# Performance settings
BATCH_SIZE=10
MAX_CONCURRENT_UPLOADS=5
```

### 7. Initialize Database

The database will be automatically created when you first run the application.

### 8. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## Configuration Options

### Custom Skill Keywords

Edit the `skills_keywords` list in the `ResumeParser` class to customize skill detection:

```python
self.skills_keywords = [
    'python', 'java', 'javascript', 'react', 'angular',
    # Add your custom skills here
]
```

### Scoring Weights

Modify scoring weights in `RelevanceEvaluator._calculate_hard_match_score()`:

```python
# Current weights: Hard Match (60%), Semantic Match (40%)
relevance_score = (hard_score * 0.6) + (semantic_score * 0.4)
```

### Verdict Thresholds

Adjust classification thresholds in `RelevanceEvaluator._determine_verdict()`:

```python
if score >= 75:      # High threshold
    return "High"
elif score >= 50:    # Medium threshold
    return "Medium"
else:
    return "Low"
```

## Troubleshooting

### Common Issues

#### 1. spaCy Model Not Found
```bash
# Error: Can't find model 'en_core_web_sm'
python -m spacy download en_core_web_sm --user
```

#### 2. Permission Errors
```bash
# On Windows (run as administrator)
pip install --user -r requirements.txt

# On macOS/Linux
sudo pip3 install -r requirements.txt
```

#### 3. Memory Issues
- Reduce batch processing size
- Process files individually
- Increase system RAM allocation

#### 4. Database Locked
```bash
# Stop all running instances and restart
pkill -f streamlit  # On macOS/Linux
taskkill /f /im python.exe  # On Windows
```

#### 5. Port Already in Use
```bash
# Run on different port
streamlit run app.py --server.port 8502
```

### Performance Optimization

#### For Large Scale Processing

1. **Increase batch size** for bulk operations:
   ```python
   # In ResumeRelevanceApp._resume_upload_page()
   BATCH_SIZE = 50  # Process 50 resumes at once
   ```

2. **Enable multiprocessing** (advanced users):
   ```python
   from multiprocessing import Pool
   # Implement parallel resume processing
   ```

3. **Database optimization**:
   ```sql
   -- Add indexes for better query performance
   CREATE INDEX idx_evaluations_score ON evaluations(relevance_score);
   CREATE INDEX idx_evaluations_job ON evaluations(job_id);
   ```

## Testing the Installation

### 1. Basic Functionality Test

1. Open the application
2. Navigate to "Upload Job Description"
3. Enter a sample job description
4. Go to "Upload Resumes"
5. Upload a test resume file
6. Check "View Results" for evaluation output

### 2. Sample Test Data

Use these sample inputs to verify functionality:

**Sample Job Description:**
```
Data Scientist - Machine Learning Engineer

We are looking for a skilled Data Scientist with 3+ years of experience in machine learning and Python programming. 

Requirements:
- Bachelor's degree in Computer Science or related field
- Strong proficiency in Python, pandas, scikit-learn
- Experience with deep learning frameworks like TensorFlow or PyTorch
- Knowledge of SQL and database management
- Experience with data visualization tools

Nice to have:
- AWS/Azure cloud experience
- Docker containerization
- Git version control
```

### 3. Validate Output

Expected system behavior:
- Job parsing extracts skills correctly
- Resume processing completes without errors
- Relevance scores are calculated (0-100 range)
- Database stores all data properly
- Analytics page shows statistics

## Directory Structure After Setup

```
resume-relevance-system/
├── app.py                     # Main application file
├── requirements.txt           # Dependencies
├── README.md                 # Project documentation
├── SETUP.md                  # This file
├── LICENSE                   # MIT license
├── .gitignore                # Git ignore rules
├── .env                      # Environment variables (optional)
├── data/                     # Database and uploads
│   └── resume_system.db      # SQLite database (auto-created)
├── logs/                     # Application logs (auto-created)
└── temp/                     # Temporary processing files (auto-created)
```

## Next Steps

1. **Customize for your needs**: Modify skill keywords and scoring logic
2. **Add test data**: Upload real job descriptions and resumes
3. **Monitor performance**: Check analytics dashboard
4. **Scale up**: Consider deployment options for production use

## Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review application logs in the `logs/` directory
3. Create an issue on GitHub with error details
4. Contact support at placement@innomatics.in

## Security Considerations

- **Data Privacy**: All resume data is stored locally in SQLite
- **File Uploads**: Only PDF/DOCX files are processed
- **Access Control**: Consider adding authentication for production deployment
- **Data Backup**: Regularly backup the SQLite database file

## Production Deployment

For production deployment, consider:

- **Docker containerization**
- **Cloud hosting** (AWS, Azure, GCP)
- **Load balancing** for multiple users
- **Database migration** to PostgreSQL/MySQL
- **Authentication system** implementation
- **SSL certificate** for HTTPS

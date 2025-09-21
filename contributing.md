# Contributing to Resume Relevance Check System

We welcome contributions to improve the Resume Relevance Check System! This document provides guidelines for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Coding Standards](#coding-standards)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

This project follows a code of conduct to ensure a welcoming environment for all contributors. Please be respectful and professional in all interactions.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and NLP concepts
- Familiarity with Streamlit framework

### Types of Contributions

We welcome several types of contributions:

1. **Bug fixes** - Fix existing issues
2. **Feature enhancements** - Improve existing functionality
3. **New features** - Add new capabilities
4. **Documentation** - Improve docs and examples
5. **Testing** - Add test cases and improve coverage
6. **Performance optimization** - Make the system faster/more efficient

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/yourusername/resume-relevance-system.git
cd resume-relevance-system

# Add the original repository as upstream
git remote add upstream https://github.com/originalowner/resume-relevance-system.git
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # On Windows: dev_env\Scripts\activate

# Install dependencies including development tools
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

### 3. Verify Setup

```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Run the application to ensure everything works
streamlit run app.py
```

## Making Changes

### 1. Create a Branch

```bash
# Update your fork
git fetch upstream
git checkout main
git merge upstream/main

# Create a feature branch
git checkout -b feature/your-feature-name
# OR
git checkout -b bugfix/issue-number
```

### 2. Development Guidelines

#### Code Organization

- Keep functions small and focused (max 50 lines)
- Use descriptive variable and function names
- Add docstrings to all functions and classes
- Follow the existing project structure

#### Key Areas for Contribution

1. **Parser Improvements** (`ResumeParser`, `JobDescriptionParser`)
   - Add support for new document formats
   - Improve skill extraction algorithms
   - Enhance text preprocessing

2. **Evaluation Logic** (`RelevanceEvaluator`)
   - Implement new matching algorithms
   - Improve scoring mechanisms
   - Add domain-specific evaluation criteria

3. **Database Operations** (`DatabaseManager`)
   - Optimize query performance
   - Add data migration utilities
   - Implement backup/restore features

4. **UI/UX Improvements**
   - Enhance Streamlit interface
   - Add new visualization components
   - Improve user experience flow

5. **Performance Optimization**
   - Implement caching mechanisms
   - Add parallel processing
   - Memory usage optimization

### 3. Commit Guidelines

Use conventional commit messages:

```bash
# Format: type(scope): description

# Examples:
git commit -m "feat(parser): add support for .txt resume files"
git commit -m "fix(database): resolve connection timeout issue"
git commit -m "docs(readme): update installation instructions"
git commit -m "test(evaluator): add unit tests for scoring algorithm"
```

**Commit Types:**
- `feat`: New features
- `fix`: Bug fixes
- `docs`: Documentation updates
- `test`: Adding tests
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `style`: Code style changes
- `chore`: Build/config updates

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_parser.py

# Run with coverage
python -m pytest --cov=src --cov-report=html
```

### Writing Tests

1. **Unit Tests** - Test individual functions
2. **Integration Tests** - Test component interactions
3. **End-to-end Tests** - Test complete workflows

```python
# Example test structure
import pytest
from src.resume_parser import ResumeParser

class TestResumeParser:
    def setup_method(self):
        self.parser = ResumeParser()
    
    def test_extract_skills(self):
        text = "I have experience with Python, Java, and machine learning"
        resume = self.parser.parse_resume(text, "test.pdf")
        assert "python" in [skill.lower() for skill in resume.skills]
        assert "java" in [skill.lower() for skill in resume.skills]
    
    def test_extract_email(self):
        text = "Contact me at john.doe@email.com for opportunities"
        email = self.parser._extract_email(text)
        assert email == "john.doe@email.com"
```

### Test Data

Use the `tests/fixtures/` directory for test data:
- Sample resumes (anonymized)
- Sample job descriptions
- Expected parsing results

## Submitting Changes

### 1. Before Submitting

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] Documentation updated if needed
- [ ] No merge conflicts with main branch
- [ ] Commit messages follow convention

### 2. Create Pull Request

```bash
# Push your changes
git push origin feature/your-feature-name

# Create PR on GitHub with:
# - Clear title and description
# - Link to related issues
# - Screenshots if UI changes
# - Test evidence if applicable
```

### 3. Pull Request Template

Use this template for your PR description:

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (specify):

## Related Issues
Closes #123, Related to #456

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing completed
- [ ] All existing tests pass

## Screenshots (if applicable)
Add screenshots for UI changes.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Coding Standards

### Python Style

Follow PEP 8 with these specifics:

```python
# Line length: 127 characters max
# Use Black for formatting
# Import organization with isort

# Function documentation
def extract_skills(self, text: str) -> List[str]:
    """Extract skills from resume text.
    
    Args:
        text: Raw resume text to process
        
    Returns:
        List of extracted skills
        
    Raises:
        ValueError: If text is empty or invalid
    """
    pass

# Class documentation
class ResumeParser:
    """Parse resume content and extract structured information.
    
    This class handles the extraction of candidate information from
    resume text, including skills, experience, and education.
    
    Attributes:
        skills_keywords: List of predefined skill keywords
        
    Example:
        parser = ResumeParser()
        resume_data = parser.parse_resume(text, filename)
    """
    pass
```

### Database Guidelines

```python
# Use context managers for database operations
def save_resume(self, resume: ResumeData):
    """Save resume data with proper error handling."""
    try:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Database operations here
            conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise
```

### Error Handling

```python
# Specific exception handling
try:
    result = risky_operation()
except SpecificException as e:
    logger.warning(f"Expected error: {e}")
    return default_value
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise

# Input validation
def parse_resume(self, text: str, filename: str) -> ResumeData:
    if not text or not text.strip():
        raise ValueError("Resume text cannot be empty")
    
    if not filename:
        raise ValueError("Filename is required")
```

### Performance Guidelines

- Use list comprehensions for simple transformations
- Implement caching for expensive operations
- Use generators for large datasets
- Profile code for bottlenecks

```python
# Good: List comprehension
skills = [skill.lower() for skill in raw_skills if skill.strip()]

# Good: Generator for memory efficiency
def process_large_dataset(items):
    for item in items:
        yield expensive_processing(item)

# Good: Caching expensive operations
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_similarity_calculation(text1, text2):
    return calculate_similarity(text1, text2)
```

## Issue Guidelines

### Reporting Bugs

Use the bug report template:

```markdown
**Bug Description**
Clear description of the bug.

**Steps to Reproduce**
1. Go to...
2. Click on...
3. See error...

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Environment**
- OS: [e.g., Windows 10, macOS 12.0]
- Python version: [e.g., 3.9.7]
- Browser: [e.g., Chrome 96.0]

**Additional Context**
Screenshots, logs, or other relevant information.
```

### Requesting Features

```markdown
**Feature Request**
Brief description of the proposed feature.

**Problem Statement**
What problem does this solve?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other approaches you've considered.

**Additional Context**
Use cases, examples, or mockups.
```

## Review Process

### For Contributors

1. **Self-review** your code before submitting
2. **Respond promptly** to review feedback
3. **Test thoroughly** after making changes
4. **Be patient** - reviews take time

### For Reviewers

1. **Be constructive** in feedback
2. **Focus on code quality** and functionality
3. **Suggest improvements** with examples
4. **Approve quickly** when ready

## Getting Help

### Resources

- **Documentation**: Check README.md and SETUP.md
- **Examples**: Look at existing code patterns
- **Issues**: Search existing issues first
- **Discussions**: Use GitHub Discussions for questions

### Contact

- Create an issue for bugs/features
- Use discussions for general questions
- Email: development@innomatics.in

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes for significant contributions
- Special mentions for major features

Thank you for contributing to the Resume Relevance Check System!

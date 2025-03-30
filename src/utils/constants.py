"""
Constants for the Resume Optimization System.
Contains configuration settings, static data, and other constants.
"""
import os

# API configuration for NLP services
API_CONFIG = {
    "api_key": os.environ.get("LLM_API_KEY", ""),  # Get API key from environment variables
    "api_url": "https://api.openai.com/v1/chat/completions",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "temperature": 0.7,
    "max_tokens": 500
}

# Skills categories and taxonomies
SKILLS_CATEGORIES = [
    "programming_languages",
    "frameworks",
    "databases",
    "cloud_services",
    "tools",
    "soft_skills"
]

# Common resume sections
RESUME_SECTIONS = [
    "contact",
    "summary",
    "experience",
    "education",
    "skills",
    "projects",
    "certifications",
    "awards",
    "publications"
]

# Job description sections
JOB_SECTIONS = [
    "title",
    "company",
    "requirements",
    "responsibilities",
    "qualifications",
    "benefits"
]

# Keywords weighting by section
SECTION_WEIGHTS = {
    "summary": 0.8,
    "experience": 1.0,
    "skills": 1.2,
    "education": 0.7,
    "projects": 0.9,
    "certifications": 0.8
}

# Resume enhancement settings
ENHANCEMENT_CONFIG = {
    "min_match_score": 0.6,  # Minimum match score to consider a section well-matched
    "max_added_keywords": 10,  # Maximum number of keywords to add in the enhancement process
    "action_verb_boost": 0.2,  # Boost for action verbs in bullet point enhancement
    "keyword_density_threshold": 0.05  # Maximum keyword density to prevent overstuffing
}

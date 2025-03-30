from pydantic import BaseModel
from typing import List, Optional

class Education(BaseModel):
    institution: str
    degree: str
    field_of_study: str
    start_date: str
    end_date: Optional[str] = None
    gpa: Optional[float] = None

class Experience(BaseModel):
    company: str
    position: str
    start_date: str
    end_date: Optional[str] = None
    description: List[str]
    skills: List[str]

class Skill(BaseModel):
    name: str
    level: Optional[str] = None
    years: Optional[float] = None

class Resume(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    summary: Optional[str] = None
    education: List[Education]
    experience: List[Experience]
    skills: List[Skill]
    certifications: Optional[List[str]] = None
    languages: Optional[List[str]] = None

class JobDescription(BaseModel):
    title: str
    company: Optional[str] = None
    description: str
    requirements: List[str]
    preferred_qualifications: Optional[List[str]] = None

class MatchResult(BaseModel):
    score: float
    matched_skills: List[str]
    missing_skills: List[str]
    recommendations: List[str]

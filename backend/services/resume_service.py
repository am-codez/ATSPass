from typing import Dict, Any
import PyPDF2
from io import BytesIO
from .nlp_service import nlp_service
from ..models.resume import Resume, JobDescription, MatchResult

class ResumeService:
    async def parse_pdf(self, file: BytesIO) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Error parsing PDF: {str(e)}")

    async def analyze_resume(self, resume_text: str) -> Dict[str, Any]:
        prompt = """
        Analyze the resume text and extract the following information in JSON format:
        {
            "name": string,
            "email": string,
            "phone": string,
            "summary": string,
            "education": [
                {
                    "institution": string,
                    "degree": string,
                    "field_of_study": string,
                    "start_date": string,
                    "end_date": string,
                    "gpa": float
                }
            ],
            "experience": [
                {
                    "company": string,
                    "position": string,
                    "start_date": string,
                    "end_date": string,
                    "description": string[],
                    "skills": string[]
                }
            ],
            "skills": [
                {
                    "name": string,
                    "level": string,
                    "years": float
                }
            ],
            "certifications": string[],
            "languages": string[]
        }
        """
        return await nlp_service.analyze_text(resume_text, prompt)

    async def match_resume_with_job(self, resume: Resume, job: JobDescription) -> MatchResult:
        resume_text = resume.model_dump_json()
        job_text = job.model_dump_json()
        
        result = await nlp_service.compare_resume_job(resume_text, job_text)
        
        return MatchResult(
            score=result.get("score", 0),
            matched_skills=result.get("matched_skills", []),
            missing_skills=result.get("missing_skills", []),
            recommendations=result.get("recommendations", [])
        )

    async def get_improvement_suggestions(self, resume: Resume, job: JobDescription) -> Dict[str, Any]:
        prompt = """
        Analyze the resume and job description, and provide specific suggestions for improving the resume.
        Focus on:
        1. Content improvements
        2. Format improvements
        3. Keyword optimization
        4. Skills to acquire
        
        Format the output as a JSON object with these categories as keys and arrays of suggestions as values.
        """
        resume_text = resume.model_dump_json()
        job_text = job.model_dump_json()
        combined_text = f"Resume:\n{resume_text}\n\nJob Description:\n{job_text}"
        
        return await nlp_service.analyze_text(combined_text, prompt)

resume_service = ResumeService()

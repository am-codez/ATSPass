from openai import OpenAI
from ..core.config import settings
from typing import List, Dict, Any
import json

class NLPService:
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.MODEL_NAME

    async def analyze_text(self, text: str, prompt: str) -> Dict[str, Any]:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes text and provides structured output."},
                    {"role": "user", "content": f"{prompt}\n\nText: {text}"}
                ],
                temperature=settings.TEMPERATURE,
                max_tokens=settings.MAX_TOKENS
            )
            
            result = response.choices[0].message.content
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"error": "Failed to parse JSON response", "raw_response": result}
                
        except Exception as e:
            return {"error": str(e)}

    async def extract_skills(self, text: str) -> List[str]:
        prompt = """
        Extract a list of technical skills, soft skills, and technologies mentioned in the text.
        Format the output as a JSON array of strings, containing only the skill names.
        Example: ["Python", "JavaScript", "Team Leadership", "Project Management"]
        """
        result = await self.analyze_text(text, prompt)
        return result.get("skills", []) if isinstance(result, dict) else result

    async def compare_resume_job(self, resume_text: str, job_description: str) -> Dict[str, Any]:
        prompt = """
        Compare the resume and job description provided. Analyze the match between them and provide:
        1. A match score (0-100)
        2. List of matching skills
        3. List of missing required skills
        4. Specific recommendations for improvement
        
        Format the output as a JSON object with the following structure:
        {
            "score": float,
            "matched_skills": string[],
            "missing_skills": string[],
            "recommendations": string[]
        }
        """
        combined_text = f"Resume:\n{resume_text}\n\nJob Description:\n{job_description}"
        return await self.analyze_text(combined_text, prompt)

nlp_service = NLPService()

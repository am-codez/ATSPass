"""
Recommendation Generator Module
Generates context-aware recommendations for resume improvement.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import json
import re
import requests
from enum import Enum
import numpy as np

# Import from other modules
from ..matching.gap_analyzer import GapAnalyzer
from ..nlp.skills_analyzer import SkillsAnalyzer
from ..utils.constants import API_CONFIG

class RecommendationType(Enum):
    """Types of recommendations that can be generated"""
    SKILL = "skill"
    EXPERIENCE = "experience"
    EDUCATION = "education"
    FORMAT = "format"
    CONTENT = "content"
    KEYWORD = "keyword"
    ATS = "ats"

class ResumeRecommender:
    """
    Generate recommendations for resume optimization
    - Context-aware suggestions
    - Industry-specific recommendations
    - ATS optimization tips
    """
    
    def __init__(self, 
                gap_analyzer: Optional[GapAnalyzer] = None,
                skills_analyzer: Optional[SkillsAnalyzer] = None,
                use_llm: bool = True,
                api_key: Optional[str] = None):
        """
        Initialize the resume recommender
        
        Args:
            gap_analyzer: Optional GapAnalyzer object
            skills_analyzer: Optional SkillsAnalyzer object
            use_llm: Whether to use LLM for enhanced suggestions
            api_key: API key for LLM service (if use_llm is True)
        """
        self.gap_analyzer = gap_analyzer or GapAnalyzer()
        self.skills_analyzer = skills_analyzer or SkillsAnalyzer()
        self.use_llm = use_llm
        self.api_key = api_key or API_CONFIG.get("api_key", "")
        self.api_url = API_CONFIG.get("api_url", "")
        
        # Load ATS optimization rules
        self.ats_rules = self._load_ats_rules()
        
        # Load industry recommendation templates
        self.industry_templates = self._load_industry_templates()
    
    def _load_ats_rules(self) -> Dict[str, Any]:
        """Load ATS optimization rules"""
        # In a real implementation, this would load from a file or database
        return {
            "format": {
                "file_types": ["PDF", "DOCX"],
                "avoid_formats": ["header/footer", "tables", "columns", "text boxes"],
                "font_recommendations": ["Arial", "Calibri", "Times New Roman"],
                "font_size": "10-12pt"
            },
            "structure": {
                "standard_sections": ["Contact Information", "Summary", "Experience", "Education", "Skills"],
                "section_order": ["Contact", "Summary", "Experience", "Education", "Skills", "Additional"]
            },
            "content": {
                "keyword_density": "3-5%",
                "avoid_graphics": True,
                "bullet_point_format": "Start with action verbs, include metrics",
                "dates_format": "MM/YYYY"
            }
        }
    
    def _load_industry_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load industry-specific recommendation templates"""
        # In a real implementation, this would load from a file or database
        return {
            "software": {
                "key_sections": ["Technical Skills", "Projects", "Experience"],
                "format_tips": "Emphasize technical skills section",
                "content_tips": "Highlight specific technologies, programming languages, and methodologies",
                "keyword_categories": ["Programming Languages", "Frameworks", "Tools", "Methodologies"]
            },
            "finance": {
                "key_sections": ["Experience", "Education", "Certifications"],
                "format_tips": "Formal structure with focus on credentials",
                "content_tips": "Emphasize quantifiable achievements and results",
                "keyword_categories": ["Analysis", "Compliance", "Reporting", "Risk Management"]
            },
            "healthcare": {
                "key_sections": ["Certifications", "Experience", "Education"],
                "format_tips": "Clear credential visibility",
                "content_tips": "Highlight patient care and specialized medical knowledge",
                "keyword_categories": ["Patient Care", "Medical Procedures", "Compliance", "Specializations"]
            },
            "marketing": {
                "key_sections": ["Experience", "Skills", "Portfolio"],
                "format_tips": "Creative but clean layout",
                "content_tips": "Focus on campaigns, metrics, and results",
                "keyword_categories": ["Campaigns", "Analytics", "Platforms", "Content Creation"]
            }
        }
    
    def get_context_aware_suggestions(self, 
                                    resume: Dict[str, Any], 
                                    job: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate context-aware suggestions based on resume and job match
        
        Args:
            resume: Resume document with sections
            job: Job document with sections
            
        Returns:
            List of suggestion objects
        """
        # Get comprehensive gap analysis
        gap_analysis = self.gap_analyzer.get_comprehensive_gap_analysis(resume, job)
        
        # Initialize suggestions list
        suggestions = []
        
        # Add skills-based suggestions
        skills_suggestions = self._generate_skills_suggestions(gap_analysis['skills_gap_analysis'])
        suggestions.extend(skills_suggestions)
        
        # Add experience-based suggestions
        experience_suggestions = self._generate_experience_suggestions(gap_analysis['experience_gap_analysis'])
        suggestions.extend(experience_suggestions)
        
        # Add education-based suggestions
        education_suggestions = self._generate_education_suggestions(gap_analysis['education_match_analysis'])
        suggestions.extend(education_suggestions)
        
        # Add format and content suggestions based on ATS rules
        ats_suggestions = self._generate_ats_suggestions(resume)
        suggestions.extend(ats_suggestions)
        
        # Enhance suggestions with LLM if enabled
        if self.use_llm:
            suggestions = self._enhance_suggestions_with_llm(suggestions, resume, job)
        
        # Sort suggestions by priority
        suggestions.sort(key=lambda x: x.get('priority', 5), reverse=True)
        
        return suggestions
    
    def _generate_skills_suggestions(self, skills_gap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on skills gap"""
        suggestions = []
        
        # Add suggestions for critical missing skills
        for skill in skills_gap.get('critical_missing_skills', []):
            suggestions.append({
                'type': RecommendationType.SKILL.value,
                'priority': 1,  # Highest priority
                'title': f"Add {skill['name']} to your resume",
                'description': f"This is a critical skill for the position that's missing from your resume",
                'details': {
                    'skill_name': skill['name'],
                    'skill_category': skill.get('category', 'Unknown'),
                    'confidence': skill.get('confidence', 1.0)
                }
            })
        
        # Add suggestions for other missing skills
        for skill in skills_gap.get('other_missing_skills', [])[:3]:  # Limit to top 3
            suggestions.append({
                'type': RecommendationType.SKILL.value,
                'priority': 3,  # Medium priority
                'title': f"Consider adding {skill['name']}",
                'description': f"This skill appears in the job description but is missing from your resume",
                'details': {
                    'skill_name': skill['name'],
                    'skill_category': skill.get('category', 'Unknown'),
                    'confidence': skill.get('confidence', 1.0)
                }
            })
        
        # Add suggestion about overall skills match
        match_percentage = skills_gap.get('match_percentage', 0)
        if match_percentage < 70:
            suggestions.append({
                'type': RecommendationType.SKILL.value,
                'priority': 2,
                'title': f"Improve skills match (currently {match_percentage:.1f}%)",
                'description': "Your resume doesn't match many of the skills in the job description",
                'details': {
                    'match_percentage': match_percentage,
                    'matching_count': skills_gap.get('total_matching_skills', 0),
                    'job_skills_count': skills_gap.get('total_job_skills', 0)
                }
            })
        
        return suggestions
    
    def _generate_experience_suggestions(self, experience_gap: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on experience gap"""
        suggestions = []
        
        # Check for years of experience gap
        experience_gap_years = experience_gap.get('experience_gap_years', 0)
        if experience_gap_years > 0:
            suggestions.append({
                'type': RecommendationType.EXPERIENCE.value,
                'priority': 2,
                'title': f"Address {experience_gap_years} year{'s' if experience_gap_years != 1 else ''} experience gap",
                'description': f"The job requires {experience_gap.get('required_years', 0)} years of experience, but your resume shows {experience_gap.get('resume_years', 0)} years",
                'details': {
                    'required_years': experience_gap.get('required_years', 0),
                    'resume_years': experience_gap.get('resume_years', 0),
                    'gap_years': experience_gap_years
                }
            })
        
        # Check for experience level compatibility
        level_compatibility = experience_gap.get('level_compatibility', 1.0)
        if level_compatibility < 0.7:
            job_seniority = experience_gap.get('job_seniority', 'mid')
            suggestions.append({
                'type': RecommendationType.EXPERIENCE.value,
                'priority': 2,
                'title': f"Highlight {job_seniority}-level experience",
                'description': f"Your resume doesn't clearly demonstrate {job_seniority}-level experience required for this role",
                'details': {
                    'compatibility_score': level_compatibility,
                    'job_level': job_seniority,
                    'resume_levels': experience_gap.get('resume_experience_level', {})
                }
            })
        
        # Check for specific experience gaps
        for phrase in experience_gap.get('experience_gap_areas', [])[:3]:  # Limit to top 3
            suggestions.append({
                'type': RecommendationType.EXPERIENCE.value,
                'priority': 3,
                'title': "Add missing experience",
                'description': f"The job requires experience in: '{phrase}'",
                'details': {
                    'missing_experience': phrase
                }
            })
        
        return suggestions
    
    def _generate_education_suggestions(self, education_match: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on education match"""
        suggestions = []
        
        # Check if education requirements are met
        if not education_match.get('meets_requirement', True):
            min_degree = education_match.get('minimum_required_degree', {}).get('name', '')
            if min_degree:
                suggestions.append({
                    'type': RecommendationType.EDUCATION.value,
                    'priority': 3,
                    'title': f"Address education requirement",
                    'description': f"The job requires a {min_degree} degree or higher",
                    'details': {
                        'required_degree': min_degree,
                        'current_degree': education_match.get('highest_resume_degree', {}).get('name', 'not specified')
                    }
                })
        
        # Check for field of study match
        field_match = education_match.get('field_match', {})
        if not field_match.get('is_match', True) and field_match.get('missing_fields'):
            fields_str = ", ".join(field_match.get('missing_fields', [])[:2])
            suggestions.append({
                'type': RecommendationType.EDUCATION.value,
                'priority': 4,
                'title': "Highlight relevant field of study",
                'description': f"The job mentions fields like {fields_str} that aren't in your education section",
                'details': {
                    'missing_fields': field_match.get('missing_fields', []),
                    'current_fields': education_match.get('resume_fields', [])
                }
            })
        
        return suggestions
    
    def _generate_ats_suggestions(self, resume: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate suggestions based on ATS optimization rules"""
        suggestions = []
        
        # Check for missing standard sections
        resume_sections = set(resume.get('sections', {}).keys())
        standard_sections = set(self.ats_rules['structure']['standard_sections'])
        missing_sections = standard_sections - resume_sections
        
        if missing_sections:
            sections_str = ", ".join(missing_sections)
            suggestions.append({
                'type': RecommendationType.ATS.value,
                'priority': 3,
                'title': "Add standard resume sections",
                'description': f"Consider adding these standard sections: {sections_str}",
                'details': {
                    'missing_sections': list(missing_sections)
                }
            })
        
        # Add general ATS format suggestions
        suggestions.append({
            'type': RecommendationType.ATS.value,
            'priority': 4,
            'title': "Optimize resume format for ATS",
            'description': "Ensure your resume uses ATS-friendly formatting",
            'details': {
                'format_tips': [
                    f"Use plain text or simple formatting",
                    f"Avoid {', '.join(self.ats_rules['format']['avoid_formats'])}",
                    f"Use standard fonts like {', '.join(self.ats_rules['format']['font_recommendations'][:2])}"
                ]
            }
        })
        
        # Add keyword placement suggestion
        suggestions.append({
            'type': RecommendationType.KEYWORD.value,
            'priority': 3,
            'title': "Optimize keyword placement",
            'description': "Place important keywords in key positions for better ATS scoring",
            'details': {
                'keyword_tips': [
                    "Include important keywords in your summary section",
                    "Match section headings to standard ATS categories",
                    "Use both acronyms and spelled-out versions of technical terms"
                ]
            }
        })
        
        return suggestions
    
    def get_industry_specific_recommendations(self, 
                                           resume: Dict[str, Any], 
                                           job: Dict[str, Any],
                                           industry: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Generate industry-specific recommendations
        
        Args:
            resume: Resume document with sections
            job: Job document with sections
            industry: Target industry (defaults to 'software' if None)
            
        Returns:
            List of industry-specific recommendation objects
        """
        # Determine industry if not provided
        if industry is None:
            industry = job.get('metadata', {}).get('industry', 'software')
        
        # Ensure industry is valid
        if industry not in self.industry_templates:
            industry = 'software'  # Default to software
        
        # Get industry template
        template = self.industry_templates[industry]
        
        # Initialize recommendations
        recommendations = []
        
        # Add industry-specific section recommendations
        resume_sections = set(resume.get('sections', {}).keys())
        industry_sections = set(template['key_sections'])
        missing_sections = industry_sections - resume_sections
        
        if missing_sections:
            sections_str = ", ".join(missing_sections)
            recommendations.append({
                'type': RecommendationType.CONTENT.value,
                'priority': 2,
                'title': f"Add key {industry} sections",
                'description': f"Consider adding these important sections for {industry} roles: {sections_str}",
                'details': {
                    'industry': industry,
                    'missing_sections': list(missing_sections)
                }
            })
        
        # Add industry format recommendation
        recommendations.append({
            'type': RecommendationType.FORMAT.value,
            'priority': 3,
            'title': f"{industry.capitalize()} format recommendation",
            'description': template['format_tips'],
            'details': {
                'industry': industry,
                'key_sections': template['key_sections']
            }
        })
        
        # Add industry content recommendation
        recommendations.append({
            'type': RecommendationType.CONTENT.value,
            'priority': 2,
            'title': f"{industry.capitalize()} content recommendation",
            'description': template['content_tips'],
            'details': {
                'industry': industry,
                'keyword_categories': template['keyword_categories']
            }
        })
        
        # Enhance with LLM if enabled
        if self.use_llm:
            recommendations = self._enhance_industry_recommendations_with_llm(recommendations, resume, job, industry)
        
        return recommendations
    
    def get_ats_optimization_tips(self, resume: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate ATS optimization tips
        
        Args:
            resume: Resume document with sections
            
        Returns:
            List of ATS optimization tip objects
        """
        # Start with standard ATS suggestions
        ats_tips = self._generate_ats_suggestions(resume)
        
        # Add more detailed ATS optimization tips
        
        # Check resume length
        resume_text = ' '.join(resume.get('sections', {}).values())
        word_count = len(resume_text.split())
        
        if word_count > 700:
            ats_tips.append({
                'type': RecommendationType.ATS.value,
                'priority': 3,
                'title': "Consider resume length",
                'description': f"Your resume is approximately {word_count} words, which may be too long for some ATS systems",
                'details': {
                    'current_words': word_count,
                    'recommended_range': "400-700 words"
                }
            })
        
        # Add file naming recommendation
        ats_tips.append({
            'type': RecommendationType.ATS.value,
            'priority': 5,  # Lower priority
            'title': "Use ATS-friendly file naming",
            'description': "Name your resume file appropriately for ATS systems",
            'details': {
                'naming_tips': [
                    "Include your name in the file name",
                    "Include the word 'resume' or 'cv'",
                    "Avoid special characters in the file name",
                    "Example: 'John_Smith_Resume.pdf'"
                ]
            }
        })
        
        # Add contact information recommendation
        ats_tips.append({
            'type': RecommendationType.ATS.value,
            'priority': 3,
            'title': "Optimize contact information",
            'description': "Ensure your contact information is ATS-friendly",
            'details': {
                'contact_tips': [
                    "Use a standard phone format like (555) 555-5555",
                    "Use a simple email format",
                    "Avoid using images or icons for contact information",
                    "Place contact information at the top of the resume"
                ]
            }
        })
        
        # Enhance with LLM if enabled
        if self.use_llm:
            ats_tips = self._enhance_ats_tips_with_llm(ats_tips, resume)
        
        return ats_tips
    
    def _enhance_suggestions_with_llm(self, 
                                    suggestions: List[Dict[str, Any]], 
                                    resume: Dict[str, Any], 
                                    job: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance suggestion descriptions using LLM"""
        if not self.api_key or not suggestions:
            return suggestions
        
        try:
            # Prepare context for LLM
            context = {
                "resume_summary": resume.get('sections', {}).get('summary', ''),
                "job_title": job.get('metadata', {}).get('title', ''),
                "job_summary": job.get('sections', {}).get('description', '')[:300],
                "suggestions": suggestions[:5]  # Limit to top 5 suggestions
            }
            
            # Create prompt for LLM
            prompt = f"""
            Given the following resume and job information:
            
            RESUME SUMMARY: {context['resume_summary']}
            
            JOB TITLE: {context['job_title']}
            JOB SUMMARY: {context['job_summary']}
            
            And these automatically generated suggestions:
            {json.dumps([s for s in context['suggestions']], indent=2)}
            
            Please enhance each suggestion description to be more specific, personalized, and actionable. Keep the same structure but make the language more natural and helpful.
            Return ONLY a JSON array with the enhanced suggestions.
            """
            
            # Call LLM API
            enhanced_suggestions = self._call_llm_api(prompt)
            
            # Process LLM response if valid
            if enhanced_suggestions and isinstance(enhanced_suggestions, list):
                # Update only the processed suggestions
                for i, suggestion in enumerate(suggestions[:5]):
                    if i < len(enhanced_suggestions):
                        # Only update description and title
                        if 'description' in enhanced_suggestions[i]:
                            suggestions[i]['description'] = enhanced_suggestions[i]['description']
                        if 'title' in enhanced_suggestions[i]:
                            suggestions[i]['title'] = enhanced_suggestions[i]['title']
            
        except Exception as e:
            print(f"Error enhancing suggestions with LLM: {str(e)}")
        
        return suggestions
    
    def _enhance_industry_recommendations_with_llm(self, 
                                                recommendations: List[Dict[str, Any]], 
                                                resume: Dict[str, Any], 
                                                job: Dict[str, Any],
                                                industry: str) -> List[Dict[str, Any]]:
        """Enhance industry-specific recommendations using LLM"""
        if not self.api_key or not recommendations:
            return recommendations
        
        try:
            # Prepare context for LLM
            context = {
                "industry": industry,
                "resume_sections": list(resume.get('sections', {}).keys()),
                "job_title": job.get('metadata', {}).get('title', ''),
                "recommendations": recommendations
            }
            
            # Create prompt for LLM
            prompt = f"""
            Given the following resume and job information for the {context['industry']} industry:
            
            JOB TITLE: {context['job_title']}
            RESUME SECTIONS: {', '.join(context['resume_sections'])}
            
            And these industry-specific recommendations:
            {json.dumps(context['recommendations'], indent=2)}
            
            Please enhance each recommendation to include specific, tailored advice for the {context['industry']} industry.
            Make the language more natural and provide actionable insights that would be valuable for someone in this field.
            Return ONLY a JSON array with the enhanced recommendations.
            """
            
            # Call LLM API
            enhanced_recommendations = self._call_llm_api(prompt)
            
            # Process LLM response if valid
            if enhanced_recommendations and isinstance(enhanced_recommendations, list):
                # Update recommendations with enhanced versions
                for i, rec in enumerate(recommendations):
                    if i < len(enhanced_recommendations):
                        # Only update description and title
                        if 'description' in enhanced_recommendations[i]:
                            recommendations[i]['description'] = enhanced_recommendations[i]['description']
                        if 'title' in enhanced_recommendations[i]:
                            recommendations[i]['title'] = enhanced_recommendations[i]['title']
            
        except Exception as e:
            print(f"Error enhancing industry recommendations with LLM: {str(e)}")
        
        return recommendations
    
    def _enhance_ats_tips_with_llm(self, 
                                 ats_tips: List[Dict[str, Any]], 
                                 resume: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhance ATS optimization tips using LLM"""
        if not self.api_key or not ats_tips:
            return ats_tips
        
        try:
            # Prepare context for LLM
            context = {
                "resume_format": resume.get('metadata', {}).get('format', 'unknown'),
                "resume_sections": list(resume.get('sections', {}).keys()),
                "ats_tips": ats_tips
            }
            
            # Create prompt for LLM
            prompt = f"""
            Given a resume with format '{context['resume_format']}' and sections {', '.join(context['resume_sections'])},
            enhance these ATS optimization tips:
            
            {json.dumps(context['ats_tips'], indent=2)}
            
            Make each tip more specific, actionable, and personalized. Provide clear examples where helpful.
            Keep ATS best practices in mind and ensure the advice helps the resume pass through ATS systems.
            Return ONLY a JSON array with the enhanced ATS tips.
            """
            
            # Call LLM API
            enhanced_tips = self._call_llm_api(prompt)
            
            # Process LLM response if valid
            if enhanced_tips and isinstance(enhanced_tips, list):
                # Update tips with enhanced versions
                for i, tip in enumerate(ats_tips):
                    if i < len(enhanced_tips):
                        # Only update description and title
                        if 'description' in enhanced_tips[i]:
                            ats_tips[i]['description'] = enhanced_tips[i]['description']
                        if 'title' in enhanced_tips[i]:
                            ats_tips[i]['title'] = enhanced_tips[i]['title']
            
        except Exception as e:
            print(f"Error enhancing ATS tips with LLM: {str(e)}")
        
        return ats_tips
    
    def _call_llm_api(self, prompt: str) -> Any:
        """
        Call LLM API with the given prompt
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Parsed JSON response or None if error
        """
        if not self.api_key or not self.api_url:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-4o-mini",  # Or your preferred model
                "messages": [
                    {"role": "system", "content": "You are a helpful resume optimization assistant. Respond only with JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            response = requests.post(self.api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                # Extract JSON from response
                response_json = response.json()
                content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # Parse JSON content
                # Find JSON array in the content
                match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str)
                
                # Try parsing the entire content as JSON
                return json.loads(content)
            
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
        
        return None

"""
Gap Analysis Module
Identifies gaps between resume and job requirements.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import re
from datetime import datetime
import numpy as np

# Import from other modules
from ..nlp.skills_analyzer import SkillsAnalyzer
from ..matching.similarity import SimilarityCalculator

class GapAnalyzer:
    """
    Analyze gaps between resume and job description
    - Skills gap detection
    - Experience gap analysis
    - Education requirements matching
    """
    
    def __init__(self, skills_analyzer: Optional[SkillsAnalyzer] = None,
                similarity_calculator: Optional[SimilarityCalculator] = None):
        """
        Initialize the gap analyzer
        
        Args:
            skills_analyzer: Optional SkillsAnalyzer object
            similarity_calculator: Optional SimilarityCalculator object
        """
        self.skills_analyzer = skills_analyzer or SkillsAnalyzer()
        self.similarity_calculator = similarity_calculator or SimilarityCalculator()
        
        # Education degree levels (ordered from lowest to highest)
        self.degree_levels = {
            "high school": 1,
            "associate": 2,
            "associate's": 2,
            "bachelor": 3,
            "bachelor's": 3,
            "bs": 3,
            "ba": 3,
            "b.s.": 3,
            "b.a.": 3,
            "master": 4,
            "master's": 4,
            "ms": 4,
            "ma": 4,
            "m.s.": 4,
            "m.a.": 4,
            "mba": 4,
            "phd": 5,
            "ph.d.": 5,
            "doctorate": 5,
            "doctoral": 5
        }
    
    def analyze_skills_gap(self, 
                         resume_text: str, 
                         job_text: str,
                         industry: str = "software") -> Dict[str, Any]:
        """
        Analyze skills gap between resume and job requirements
        
        Args:
            resume_text: Full resume text
            job_text: Full job description text
            industry: Industry for skill relevance
            
        Returns:
            Dictionary with skills gap analysis
        """
        # Extract skills from resume and job description
        resume_skills = self.skills_analyzer.extract_skills(resume_text)
        job_skills = self.skills_analyzer.extract_skills(job_text)
        
        # Get industry-specific skills
        industry_skills = self.skills_analyzer.get_industry_specific_skills(industry)
        
        # Create sets of skill names for comparison
        # Handle both string skills and dictionary skills
        resume_skill_names = set()
        for skill in resume_skills:
            if isinstance(skill, dict):
                if 'name' in skill:
                    resume_skill_names.add(skill['name'].lower())
                elif 'skill' in skill:
                    resume_skill_names.add(skill['skill'].lower())
            elif isinstance(skill, str):
                resume_skill_names.add(skill.lower())
        
        job_skill_names = set()
        for skill in job_skills:
            if isinstance(skill, dict):
                if 'name' in skill:
                    job_skill_names.add(skill['name'].lower())
                elif 'skill' in skill:
                    job_skill_names.add(skill['skill'].lower())
            elif isinstance(skill, str):
                job_skill_names.add(skill.lower())
        
        # Identify matching and missing skills
        matching_skills = resume_skill_names.intersection(job_skill_names)
        missing_skills = job_skill_names - resume_skill_names
        
        # Calculate match percentage
        match_percentage = len(matching_skills) / len(job_skill_names) * 100 if job_skill_names else 0
        
        # Extract details of missing skills
        missing_skill_details = []
        for skill in job_skills:
            skill_name = ""
            if isinstance(skill, dict):
                if 'name' in skill:
                    skill_name = skill['name'].lower()
                elif 'skill' in skill:
                    skill_name = skill['skill'].lower()
            elif isinstance(skill, str):
                skill_name = skill.lower()
                
            if skill_name in missing_skills:
                # Convert string skills to dictionary format
                if isinstance(skill, str):
                    missing_skill_details.append({
                        'name': skill,
                        'category': 'Uncategorized',
                        'confidence': 1.0
                    })
                else:
                    # Make sure we have a 'name' key for standardization
                    if 'name' not in skill and 'skill' in skill:
                        skill_copy = skill.copy()
                        skill_copy['name'] = skill_copy['skill']
                        missing_skill_details.append(skill_copy)
                    else:
                        missing_skill_details.append(skill)
        
        # Sort missing skills by importance in the industry
        industry_skill_names = {skill.lower() for skill in industry_skills}
        critical_missing = []
        other_missing = []
        
        for skill in missing_skill_details:
            skill_name = skill.get('name', '').lower()
            if skill_name in industry_skill_names:
                critical_missing.append(skill)
            else:
                other_missing.append(skill)
        
        # Categorize missing skills
        categorized_missing = self.skills_analyzer.classify_skills(missing_skill_details)
        
        return {
            'matching_skills': list(matching_skills),
            'missing_skills': list(missing_skills),
            'critical_missing_skills': critical_missing,
            'other_missing_skills': other_missing,
            'categorized_missing_skills': categorized_missing,
            'match_percentage': match_percentage,
            'total_job_skills': len(job_skill_names),
            'total_resume_skills': len(resume_skill_names),
            'total_matching_skills': len(matching_skills)
        }
    
    def analyze_experience_gap(self, 
                             resume_experience: str, 
                             job_requirements: str) -> Dict[str, Any]:
        """
        Analyze experience gap between resume and job requirements
        
        Args:
            resume_experience: Experience section from resume
            job_requirements: Requirements section from job description
            
        Returns:
            Dictionary with experience gap analysis
        """
        # Extract years of experience from resume
        resume_years = self._extract_years_of_experience(resume_experience)
        
        # Extract required years from job description
        required_years = self._extract_years_of_experience(job_requirements)
        
        # Calculate experience gap
        experience_gap = required_years - resume_years if required_years > resume_years else 0
        
        # Determine experience level from resume
        resume_experience_level = self.skills_analyzer.detect_experience_level(resume_experience)
        
        # Extract job seniority level
        job_seniority = self._extract_job_seniority(job_requirements)
        
        # Calculate compatibility score
        level_compatibility = self._calculate_level_compatibility(
            resume_experience_level, 
            job_seniority
        )
        
        # Identify key missing experience elements using semantic analysis
        key_job_phrases = self._extract_key_experience_phrases(job_requirements)
        matching_experiences = self._match_experience_phrases(resume_experience, key_job_phrases)
        
        # Non-matching phrases represent experience gaps
        experience_gap_phrases = [
            phrase for phrase in key_job_phrases
            if phrase not in [match['phrase'] for match in matching_experiences]
        ]
        
        return {
            'resume_years': resume_years,
            'required_years': required_years,
            'experience_gap_years': experience_gap,
            'resume_experience_level': resume_experience_level,
            'job_seniority': job_seniority,
            'level_compatibility': level_compatibility,
            'experience_gap_areas': experience_gap_phrases,
            'matching_experiences': matching_experiences
        }
    
    def analyze_education_match(self, 
                              resume_education: str, 
                              job_requirements: str) -> Dict[str, Any]:
        """
        Analyze education requirements matching
        
        Args:
            resume_education: Education section from resume
            job_requirements: Requirements section from job description
            
        Returns:
            Dictionary with education match analysis
        """
        # Extract degrees from resume
        resume_degrees = self._extract_degrees(resume_education)
        
        # Extract required degrees from job description
        required_degrees = self._extract_degrees(job_requirements)
        
        # Find highest degree level in resume
        highest_resume_degree = self._find_highest_degree(resume_degrees)
        
        # Find minimum required degree in job description
        minimum_required_degree = self._find_minimum_degree(required_degrees)
        
        # Check if resume meets the minimum requirement
        meets_requirement = self._check_degree_requirement(
            highest_resume_degree, 
            minimum_required_degree
        )
        
        # Check for field of study match
        resume_fields = self._extract_fields_of_study(resume_education)
        required_fields = self._extract_fields_of_study(job_requirements)
        
        field_match = self._check_field_match(resume_fields, required_fields)
        
        # Calculate education match score
        education_match_score = self._calculate_education_match_score(
            meets_requirement,
            field_match
        )
        
        return {
            'resume_degrees': resume_degrees,
            'required_degrees': required_degrees,
            'highest_resume_degree': highest_resume_degree,
            'minimum_required_degree': minimum_required_degree,
            'meets_requirement': meets_requirement,
            'resume_fields': resume_fields,
            'required_fields': required_fields,
            'field_match': field_match,
            'education_match_score': education_match_score
        }
    
    def get_comprehensive_gap_analysis(self, 
                                     resume: Dict[str, Any], 
                                     job: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive gap analysis between resume and job
        
        Args:
            resume: Resume document with sections
            job: Job document with sections
            
        Returns:
            Dictionary with comprehensive gap analysis
        """
        # Extract relevant sections
        resume_experience = resume.get('sections', {}).get('experience', '')
        resume_education = resume.get('sections', {}).get('education', '')
        resume_skills = resume.get('sections', {}).get('skills', '')
        resume_full_text = ' '.join(resume.get('sections', {}).values())
        
        job_requirements = job.get('sections', {}).get('requirements', '')
        job_description = job.get('sections', {}).get('description', '')
        job_full_text = ' '.join(job.get('sections', {}).values())
        
        # Analyze skills gap
        skills_analysis = self.analyze_skills_gap(
            resume_full_text, 
            job_full_text,
            job.get('metadata', {}).get('industry', 'software')
        )
        
        # Analyze experience gap
        experience_analysis = self.analyze_experience_gap(
            resume_experience, 
            job_requirements
        )
        
        # Analyze education match
        education_analysis = self.analyze_education_match(
            resume_education, 
            job_requirements
        )
        
        # Calculate overall match score
        overall_match = self._calculate_overall_match(
            skills_analysis,
            experience_analysis,
            education_analysis
        )
        
        # Get key improvement areas
        improvement_areas = self._identify_improvement_areas(
            skills_analysis,
            experience_analysis,
            education_analysis
        )
        
        return {
            'skills_gap_analysis': skills_analysis,
            'experience_gap_analysis': experience_analysis,
            'education_match_analysis': education_analysis,
            'overall_match_score': overall_match,
            'key_improvement_areas': improvement_areas
        }
    
    def _extract_years_of_experience(self, text: str) -> float:
        """Extract years of experience from text"""
        # Look for patterns like "X years of experience"
        year_patterns = [
            r'(\d+)\+?\s*(?:years|yrs)(?:\s*of)?\s*(?:experience|exp)',
            r'(?:experience|exp)(?:\s*of)?\s*(\d+)\+?\s*(?:years|yrs)',
            r'(\d+)\+?\s*(?:years|yrs)'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                # Return the highest number of years found
                return max(float(year) for year in matches)
        
        # Default if no match found
        return 0.0
    
    def _extract_job_seniority(self, text: str) -> str:
        """Extract job seniority level from text"""
        # Define seniority level patterns
        seniority_patterns = {
            'entry': r'\b(?:entry|junior|jr|beginner)\b',
            'mid': r'\b(?:mid|intermediate|regular)\b',
            'senior': r'\b(?:senior|sr|experienced|lead)\b',
            'expert': r'\b(?:expert|principal|staff|director)\b'
        }
        
        # Check for each seniority level
        for level, pattern in seniority_patterns.items():
            if re.search(pattern, text.lower()):
                return level
        
        # Default to mid-level if no specific level found
        return 'mid'
    
    def _calculate_level_compatibility(self, 
                                     resume_levels: Dict[str, float], 
                                     job_level: str) -> float:
        """Calculate compatibility between resume experience level and job seniority"""
        # Job level to expected resume level mapping
        job_to_resume_mapping = {
            'entry': 'entry',
            'mid': 'mid',
            'senior': 'senior',
            'expert': 'expert'
        }
        
        # Get expected level for the job
        expected_level = job_to_resume_mapping.get(job_level, 'mid')
        
        # Check resume confidence for that level
        compatibility = resume_levels.get(expected_level, 0.0)
        
        # If job requires higher level than detected in resume
        level_order = ['entry', 'mid', 'senior', 'expert']
        job_index = level_order.index(job_level)
        
        # Find highest resume level with confidence > 0.5
        highest_resume_level = 'entry'
        for level in level_order:
            if resume_levels.get(level, 0.0) > 0.5:
                highest_resume_level = level
        
        highest_index = level_order.index(highest_resume_level)
        
        # If resume level is lower than job level, reduce compatibility
        if highest_index < job_index:
            compatibility *= 0.5  # Penalty for under-qualification
        
        return min(compatibility, 1.0)  # Cap at 1.0
    
    def _extract_key_experience_phrases(self, text: str) -> List[str]:
        """Extract key experience phrases from job requirements"""
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        # Extract phrases that indicate required experience
        experience_phrases = []
        
        for sentence in sentences:
            # Skip sentences that don't have experience indicators
            if not re.search(r'\b(?:experience|background|skill|proficiency|knowledge)\b', sentence.lower()):
                continue
                
            # Clean up sentence
            cleaned = re.sub(r'\s+', ' ', sentence).strip()
            experience_phrases.append(cleaned)
        
        return experience_phrases
    
    def _match_experience_phrases(self, 
                               resume_text: str, 
                               job_phrases: List[str]) -> List[Dict[str, Any]]:
        """Match job experience phrases to resume content"""
        matches = []
        
        for phrase in job_phrases:
            # Calculate semantic similarity to resume
            similarity = self.similarity_calculator.semantic_similarity(phrase, resume_text)
            
            # Consider as a match if similarity is above threshold
            if similarity > 0.6:  # Threshold determined empirically
                matches.append({
                    'phrase': phrase,
                    'similarity': similarity,
                    'is_match': True
                })
            else:
                matches.append({
                    'phrase': phrase,
                    'similarity': similarity,
                    'is_match': False
                })
        
        # Sort by similarity (highest first)
        return sorted(matches, key=lambda x: x['similarity'], reverse=True)
    
    def _extract_degrees(self, text: str) -> List[str]:
        """Extract degree mentions from text"""
        degree_pattern = r'\b(?:' + '|'.join(self.degree_levels.keys()) + r')\b'
        matches = re.findall(degree_pattern, text.lower())
        return matches
    
    def _find_highest_degree(self, degrees: List[str]) -> Dict[str, Any]:
        """Find highest degree mentioned in a list"""
        if not degrees:
            return {'name': 'none', 'level': 0}
            
        highest_level = 0
        highest_name = 'none'
        
        for degree in degrees:
            level = self.degree_levels.get(degree.lower(), 0)
            if level > highest_level:
                highest_level = level
                highest_name = degree
        
        return {'name': highest_name, 'level': highest_level}
    
    def _find_minimum_degree(self, degrees: List[str]) -> Dict[str, Any]:
        """Find minimum required degree from a list"""
        if not degrees:
            return {'name': 'none', 'level': 0}
            
        # Start with highest possible level
        min_level = 6
        min_name = 'none'
        
        for degree in degrees:
            level = self.degree_levels.get(degree.lower(), 0)
            if 0 < level < min_level:  # Only consider valid degrees
                min_level = level
                min_name = degree
        
        return {'name': min_name, 'level': min_level}
    
    def _check_degree_requirement(self, 
                               resume_degree: Dict[str, Any], 
                               required_degree: Dict[str, Any]) -> bool:
        """Check if resume meets degree requirement"""
        return resume_degree['level'] >= required_degree['level']
    
    def _extract_fields_of_study(self, text: str) -> List[str]:
        """Extract fields of study from text"""
        # Common fields of study
        fields = [
            'computer science', 'engineering', 'business', 'finance', 'marketing',
            'mathematics', 'statistics', 'physics', 'chemistry', 'biology',
            'psychology', 'economics', 'accounting', 'communications', 'education',
            'information technology', 'data science', 'machine learning',
            'artificial intelligence', 'electrical engineering', 'mechanical engineering',
            'civil engineering', 'chemical engineering', 'computer engineering'
        ]
        
        found_fields = []
        for field in fields:
            if re.search(r'\b' + re.escape(field) + r'\b', text.lower()):
                found_fields.append(field)
        
        return found_fields
    
    def _check_field_match(self, 
                         resume_fields: List[str], 
                         required_fields: List[str]) -> Dict[str, Any]:
        """Check if resume fields match required fields"""
        if not required_fields:
            return {'is_match': True, 'matching_fields': [], 'missing_fields': []}
            
        matching_fields = [field for field in resume_fields if field in required_fields]
        missing_fields = [field for field in required_fields if field not in resume_fields]
        
        is_match = len(matching_fields) > 0
        
        return {
            'is_match': is_match,
            'matching_fields': matching_fields,
            'missing_fields': missing_fields
        }
    
    def _calculate_education_match_score(self, 
                                       meets_requirement: bool, 
                                       field_match: Dict[str, Any]) -> float:
        """Calculate education match score"""
        base_score = 0.6 if meets_requirement else 0.2
        
        # Add points for field match
        if field_match['is_match']:
            # More points if multiple fields match
            field_bonus = min(0.4, len(field_match['matching_fields']) * 0.2)
            return min(1.0, base_score + field_bonus)
        
        return base_score
    
    def _calculate_overall_match(self, 
                               skills_analysis: Dict[str, Any],
                               experience_analysis: Dict[str, Any],
                               education_analysis: Dict[str, Any]) -> float:
        """Calculate overall match score"""
        # Extract individual scores
        skills_score = skills_analysis['match_percentage'] / 100
        
        experience_score = 1.0 - min(1.0, experience_analysis['experience_gap_years'] / 5)
        experience_score = experience_score * experience_analysis['level_compatibility']
        
        education_score = education_analysis['education_match_score']
        
        # Calculate weighted sum
        weights = {
            'skills': 0.5,
            'experience': 0.3,
            'education': 0.2
        }
        
        overall_score = (
            skills_score * weights['skills'] +
            experience_score * weights['experience'] +
            education_score * weights['education']
        )
        
        return overall_score
    
    def _identify_improvement_areas(self, 
                                  skills_analysis: Dict[str, Any],
                                  experience_analysis: Dict[str, Any],
                                  education_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify key areas for improvement"""
        improvement_areas = []
        
        # Add critical missing skills
        if skills_analysis['critical_missing_skills']:
            improvement_areas.append({
                'category': 'skills',
                'importance': 'high',
                'description': 'Add critical missing skills',
                'details': [skill['name'] for skill in skills_analysis['critical_missing_skills']]
            })
        
        # Check for experience gaps
        if experience_analysis['experience_gap_years'] > 0:
            improvement_areas.append({
                'category': 'experience',
                'importance': 'high',
                'description': f"Address {experience_analysis['experience_gap_years']} years experience gap",
                'details': [f"Need {experience_analysis['required_years']} years, have {experience_analysis['resume_years']} years"]
            })
        
        # Check for specific experience areas
        if experience_analysis['experience_gap_areas']:
            improvement_areas.append({
                'category': 'experience',
                'importance': 'medium',
                'description': 'Address specific experience requirements',
                'details': experience_analysis['experience_gap_areas']
            })
        
        # Check for education requirements
        if not education_analysis['meets_requirement']:
            improvement_areas.append({
                'category': 'education',
                'importance': 'medium',
                'description': 'Address education requirements',
                'details': [f"Need {education_analysis['minimum_required_degree']['name']} degree or higher"]
            })
        
        # Check for field of study match
        if not education_analysis['field_match']['is_match'] and education_analysis['field_match']['missing_fields']:
            improvement_areas.append({
                'category': 'education',
                'importance': 'low',
                'description': 'Address field of study requirements',
                'details': education_analysis['field_match']['missing_fields']
            })
        
        # Sort by importance
        importance_order = {'high': 0, 'medium': 1, 'low': 2}
        improvement_areas.sort(key=lambda x: importance_order[x['importance']])
        
        return improvement_areas

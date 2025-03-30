from typing import Dict, List, Set, Tuple
import re
import json
from src.preprocessing.cleaner import TextCleaner
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

"""
skills_analyzer.py:
     - Skills taxonomy implementation
     - Industry-specific skills mapping
     - Experience level detection
"""

class SkillsAnalyzer:
    def __init__(self, industry: str = "software"):
        """
        Initialize the SkillsAnalyzer
        
        Args:
            industry: The industry to focus on for skills analysis
        """
        self.industry = industry
        
        # Initialize NLP components
        self.cleaner = TextCleaner()
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        
        # Load skills taxonomy and mappings
        self.skills_taxonomy = self._load_skills_taxonomy()
        self.industry_skills_map = self._load_industry_skills_map()
        
        # Experience level indicators
        self.experience_levels = {
            'entry': {
                'years': [0, 1, 2],
                'keywords': ['junior', 'entry', 'entry-level', 'intern', 'internship', 'trainee', 'assistant'],
                'verbs': ['assisted', 'learned', 'trained', 'supported', 'helped']
            },
            'mid': {
                'years': [2, 3, 4, 5],
                'keywords': ['mid-level', 'intermediate', 'associate'],
                'verbs': ['implemented', 'developed', 'built', 'created', 'designed', 'tested']
            },
            'senior': {
                'years': [5, 6, 7, 8, 9, 10],
                'keywords': ['senior', 'lead', 'sr', 'experienced', 'specialist'],
                'verbs': ['led', 'managed', 'architected', 'mentored', 'oversaw', 'directed', 'coordinated']
            },
            'expert': {
                'years': [10, 15, 20],
                'keywords': ['principal', 'staff', 'expert', 'architect', 'director', 'head', 'chief'],
                'verbs': ['strategized', 'spearheaded', 'pioneered', 'innovated', 'transformed', 'optimized']
            }
        }

    def _load_skills_taxonomy(self) -> Dict[str, Dict]:
        """
        Load skills taxonomy - categorized skills hierarchy
        
        In a real implementation, this would load from a database or file
        """
        taxonomy = {
            # Programming Languages
            "programming_languages": {
                "name": "Programming Languages",
                "skills": {
                    "python": {
                        "name": "Python",
                        "related": ["django", "flask", "pandas", "numpy", "tensorflow", "pytorch"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "java": {
                        "name": "Java",
                        "related": ["spring", "hibernate", "maven", "gradle", "junit"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "javascript": {
                        "name": "JavaScript",
                        "related": ["react", "angular", "vue", "node", "express", "jquery"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "typescript": {
                        "name": "TypeScript",
                        "related": ["angular", "react", "node"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "c++": {
                        "name": "C++",
                        "related": ["stl", "boost", "cmake", "unreal"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    }
                }
            },
            
            # Web Development
            "web_development": {
                "name": "Web Development",
                "skills": {
                    "frontend": {
                        "name": "Frontend Development",
                        "related": ["html", "css", "javascript", "react", "angular", "vue", "redux", "webpack"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "backend": {
                        "name": "Backend Development",
                        "related": ["node", "express", "django", "flask", "spring", "ruby on rails", "php", "laravel"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "fullstack": {
                        "name": "Full Stack Development",
                        "related": ["frontend", "backend", "api", "database"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    }
                }
            },
            
            # Cloud & DevOps
            "cloud_devops": {
                "name": "Cloud & DevOps",
                "skills": {
                    "aws": {
                        "name": "Amazon Web Services",
                        "related": ["ec2", "s3", "lambda", "cloudformation", "rds", "dynamodb"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "azure": {
                        "name": "Microsoft Azure",
                        "related": ["azure functions", "cosmos db", "azure devops", "app service"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "gcp": {
                        "name": "Google Cloud Platform",
                        "related": ["compute engine", "cloud functions", "bigquery", "cloud storage"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "docker": {
                        "name": "Docker",
                        "related": ["containerization", "docker-compose", "dockerfile"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "kubernetes": {
                        "name": "Kubernetes",
                        "related": ["k8s", "container orchestration", "helm", "kubectl"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    }
                }
            },
            
            # Data Science
            "data_science": {
                "name": "Data Science",
                "skills": {
                    "machine_learning": {
                        "name": "Machine Learning",
                        "related": ["classification", "regression", "clustering", "neural networks"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "deep_learning": {
                        "name": "Deep Learning",
                        "related": ["neural networks", "cnn", "rnn", "lstm", "transformer"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "data_analysis": {
                        "name": "Data Analysis",
                        "related": ["pandas", "numpy", "statistics", "visualization", "tableau", "power bi"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "nlp": {
                        "name": "Natural Language Processing",
                        "related": ["transformers", "bert", "gpt", "spacy", "nltk", "word embeddings"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    }
                }
            },
            
            # Databases
            "databases": {
                "name": "Databases",
                "skills": {
                    "sql": {
                        "name": "SQL",
                        "related": ["postgresql", "mysql", "oracle", "sql server", "queries", "joins"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "nosql": {
                        "name": "NoSQL",
                        "related": ["mongodb", "cassandra", "redis", "dynamodb", "couchbase"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "graph_db": {
                        "name": "Graph Databases",
                        "related": ["neo4j", "tigergraph", "arangodb", "janusgraph"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    }
                }
            },
            
            # Soft Skills
            "soft_skills": {
                "name": "Soft Skills",
                "skills": {
                    "communication": {
                        "name": "Communication",
                        "related": ["writing", "speaking", "presenting", "listening", "interpersonal"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "leadership": {
                        "name": "Leadership",
                        "related": ["team management", "mentoring", "motivation", "delegation"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    },
                    "problem_solving": {
                        "name": "Problem Solving",
                        "related": ["critical thinking", "analytical", "creative thinking", "decision making"],
                        "level": ["beginner", "intermediate", "advanced", "expert"]
                    }
                }
            }
        }
        
        return taxonomy

    def _load_industry_skills_map(self) -> Dict[str, List[str]]:
        """
        Load industry-specific skills mapping
        
        In a real implementation, this would load from a database or file
        """
        industry_maps = {
            "software": [
                "programming_languages", "web_development", "cloud_devops", "databases", "soft_skills"
            ],
            "data_science": [
                "programming_languages", "data_science", "databases", "soft_skills"
            ],
            "healthcare": [
                "healthcare_systems", "medical_terminology", "patient_care", "healthcare_compliance", "soft_skills"
            ],
            "finance": [
                "financial_analysis", "investment", "risk_management", "accounting", "soft_skills"
            ]
        }
        
        return industry_maps.get(self.industry.lower(), ["soft_skills"])

    def extract_skills(self, text: str) -> List[Dict]:
        """
        Extract skills from text using the taxonomy
        
        Args:
            text: Text to extract skills from
            
        Returns:
            List of detected skills with categories and confidence scores
        """
        # Clean and prepare the text
        clean_text = self.cleaner.clean_text(text)
        clean_text = clean_text.lower()
        
        # Tokenize
        tokens = word_tokenize(clean_text)
        
        # Create doc for NER
        doc = self.nlp(clean_text)
        
        extracted_skills = []
        processed_skills = set()
        
        # Extract skills from taxonomy categories relevant to the industry
        for category_key in self.industry_skills_map:
            if category_key in self.skills_taxonomy:
                category = self.skills_taxonomy[category_key]
                category_name = category["name"]
                
                for skill_key, skill_data in category["skills"].items():
                    skill_name = skill_data["name"].lower()
                    related_terms = [term.lower() for term in skill_data["related"]]
                    
                    # Check for explicit mention of the skill
                    if skill_key in clean_text or skill_name in clean_text:
                        if skill_key not in processed_skills:
                            extracted_skills.append({
                                "skill": skill_name,
                                "category": category_name,
                                "confidence": 0.9,
                                "matches": [skill_name]
                            })
                            processed_skills.add(skill_key)
                    
                    # Check for related terms
                    matches = []
                    for term in related_terms:
                        if term in clean_text:
                            matches.append(term)
                    
                    if matches and skill_key not in processed_skills:
                        extracted_skills.append({
                            "skill": skill_name,
                            "category": category_name,
                            "confidence": 0.7,
                            "matches": matches
                        })
                        processed_skills.add(skill_key)
        
        return extracted_skills

    def classify_skills(self, skills: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Classify extracted skills by category
        
        Args:
            skills: List of extracted skills
            
        Returns:
            Dictionary of skills organized by category
        """
        classified = {}
        
        for skill in skills:
            # Handle different dictionary formats
            if 'category' in skill:
                category = skill["category"]
            else:
                category = "Uncategorized"
                
            if category not in classified:
                classified[category] = []
            
            classified[category].append(skill)
        
        return classified

    def detect_experience_level(self, text: str) -> Dict[str, float]:
        """
        Detect experience level from text
        
        Args:
            text: Text to analyze for experience level
            
        Returns:
            Dictionary with confidence scores for each experience level
        """
        # Clean and prepare the text
        clean_text = self.cleaner.clean_text(text)
        clean_text = clean_text.lower()
        
        # Create spaCy doc
        doc = self.nlp(clean_text)
        
        # Initialize confidence scores
        scores = {
            'entry': 0.0,
            'mid': 0.0,
            'senior': 0.0,
            'expert': 0.0
        }
        
        # Check for years of experience
        year_patterns = [
            r'(\d+)\+?\s*years?(?:\s*of)?\s*experience',
            r'experience\s*(?:of|for)?\s*(\d+)\+?\s*years?',
            r'(?:with)?\s*(\d+)\+?\s*years?(?:\s*of)?\s*experience'
        ]
        
        max_years = 0
        for pattern in year_patterns:
            matches = re.finditer(pattern, clean_text)
            for match in matches:
                years = int(match.group(1))
                if years > max_years:
                    max_years = years
        
        # Assign experience level based on years
        if max_years > 0:
            for level, data in self.experience_levels.items():
                if max_years in data['years']:
                    scores[level] += 0.6
                    
        # Check for level keywords
        for level, data in self.experience_levels.items():
            for keyword in data['keywords']:
                if keyword in clean_text:
                    scores[level] += 0.4
            
            # Check for action verbs
            verb_count = 0
            for verb in data['verbs']:
                if verb in clean_text:
                    verb_count += 1
            
            if verb_count > 0:
                # Normalize based on number of verbs found
                verb_score = min(0.3, 0.1 * verb_count)
                scores[level] += verb_score
        
        # Normalize scores
        total = sum(scores.values())
        if total > 0:
            for level in scores:
                scores[level] /= total
                
        # If no clear indicators found, default to mid-level
        if max(scores.values()) < 0.1:
            scores['mid'] = 1.0
            scores['entry'] = 0.0
            scores['senior'] = 0.0
            scores['expert'] = 0.0
        
        return scores
    
    def get_skills_with_experience(self, text: str) -> Dict:
        """
        Get skills with detected experience level
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with skills and experience level
        """
        # Extract skills
        skills = self.extract_skills(text)
        
        # Detect experience level
        experience = self.detect_experience_level(text)
        
        # Classify skills
        classified_skills = self.classify_skills(skills)
        
        # Determine primary experience level
        primary_level = max(experience.items(), key=lambda x: x[1])[0]
        
        return {
            "skills": classified_skills,
            "experience_level": {
                "primary": primary_level,
                "confidence_scores": experience
            },
            "industry": self.industry
        }
    
    def get_industry_specific_skills(self, skills_or_industry):
        """
        Filter skills that are particularly important for the selected industry
        
        Args:
            skills_or_industry: Either a list of extracted skills or an industry name string
            
        Returns:
            List of industry-relevant skills with importance scores or list of relevant skills for industry
        """
        # Check if input is an industry name string
        if isinstance(skills_or_industry, str):
            # Return a list of important skills for this industry
            industry = skills_or_industry.lower()
            industry_skills = []
            
            if industry in self.industry_skills_map:
                for category_key in self.industry_skills_map:
                    if category_key in self.skills_taxonomy:
                        category = self.skills_taxonomy[category_key]
                        for skill_key, skill_data in category["skills"].items():
                            skill_name = skill_data["name"].lower()
                            industry_skills.append(skill_name)
            
            return industry_skills
        
        # Original implementation for list of skills input
        skills = skills_or_industry
        industry_specific = []
        
        # Define industry importance multipliers
        industry_importance = {
            "software": {
                "Programming Languages": 1.5,
                "Web Development": 1.3,
                "Cloud & DevOps": 1.4,
                "Databases": 1.2,
                "Soft Skills": 1.0
            },
            "data_science": {
                "Programming Languages": 1.3,
                "Data Science": 1.5,
                "Databases": 1.2,
                "Soft Skills": 1.0
            }
        }
        
        # Get importance multipliers for current industry
        multipliers = industry_importance.get(self.industry.lower(), {})
        
        for skill in skills:
            category = skill["category"]
            importance = multipliers.get(category, 1.0)
            
            # Copy the skill and add importance score
            enhanced_skill = skill.copy()
            enhanced_skill["industry_importance"] = importance
            
            industry_specific.append(enhanced_skill)
            
        # Sort by industry importance
        industry_specific.sort(key=lambda x: x["industry_importance"], reverse=True)
        
        return industry_specific

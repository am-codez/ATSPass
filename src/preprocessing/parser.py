"""
Document Parser Module
Handles extraction and preprocessing of resume and job description content.
"""

import PyPDF2
from docx import Document
import spacy
from typing import Dict, List, Optional
import re

class DocumentParser:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def parse_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfFileReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text

    def parse_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text
        return text

class ResumeParser(DocumentParser):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> Dict[str, str]:
        """
        Parse resume text and extract structured sections
        
        Args:
            text: Raw resume text
            
        Returns:
            Dictionary with resume sections
        """
        # Extract all sections
        sections = self.extract_sections(text)
        
        # If we couldn't identify proper sections, create a basic structure
        if not sections:
            # Try to extract contact info
            contact_info = self.extract_contact_info(text)
            
            # Create a basic structure with the whole text as "content"
            sections = {
                'content': text,
                'contact': ', '.join([v for k, v in contact_info.items() if v])
            }
        
        return {'sections': sections}

    # Gets just the text from the resume
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Identify and extract different sections"""
        # Define common section headers with variations
        section_mapping = {
            'education': [
                r'education',
                r'academic background',
                r'academic history',
                r'degrees',
                r'qualifications'
            ],
            'experience': [
                r'experience',
                r'work experience',
                r'employment history',
                r'professional experience',
                r'work history',
                r'career history'
            ],
            'skills': [
                r'skills',
                r'technical skills',
                r'core competencies',
                r'proficiencies',
                r'expertise',
                r'key skills',
                r'technical expertise'
            ],
            'projects': [
                r'projects',
                r'personal projects',
                r'academic projects',
                r'professional projects',
                r'key projects'
            ],
            'certifications': [
                r'certifications',
                r'certificates',
                r'professional certifications',
                r'licenses'
            ],
            'publications': [
                r'publications',
                r'research publications',
                r'papers',
                r'articles'
            ],
            'awards': [
                r'awards',
                r'honors',
                r'achievements',
                r'recognitions'
            ],
            'references': [
                r'references',
                r'professional references'
            ],
            'volunteer': [
                r'volunteer',
                r'volunteering',
                r'volunteer experience',
                r'community service'
            ],
            'extracurricular': [
                r'extracurricular',
                r'activities',
                r'extracurricular activities',
                r'hobbies',
                r'interests'
            ],
            'leadership': [
                r'leadership',
                r'leadership experience',
                r'leadership roles'
            ],
            'summary': [
                r'summary',
                r'professional summary',
                r'executive summary',
                r'profile',
                r'objective',
                r'about me'
            ]
        }
        
        # Compile regex patterns
        section_patterns = {}
        for section, variants in section_mapping.items():
            patterns = [fr'\b{variant}\b\s*:?' for variant in variants]
            section_patterns[section] = re.compile('|'.join(patterns), re.IGNORECASE)
        
        # Initialize results
        sections = {}
        current_section = None
        current_content = []
        
        # Split text into lines
        lines = text.split('\n')
        
        for line in lines:
            # Check if line is a section header
            found_section = None
            for section, pattern in section_patterns.items():
                if pattern.search(line):
                    found_section = section
                    break
            
            if found_section:
                # Save previous section if it exists
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = found_section
                current_content = []
            elif current_section and line.strip():  # Add non-empty lines to current section
                current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information using NER"""
        # Use spaCy NER for names, locations
        doc = self.nlp(text)
        
        # Extract email
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        emails = re.findall(email_pattern, text)
        
        # Extract phone
        phone_pattern = r'(\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}'
        phones = re.findall(phone_pattern, text)
        
        # Extract name using NER
        name = None
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                name = ent.text
                break
        
        # Extract location
        location = None
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                location = ent.text
                break
        
        return {
            'name': name,
            'email': emails[0] if emails else None,
            'phone': phones[0] if phones else None,
            'location': location
        }

    def extract_education(self, text: str) -> List[Dict]:
        """Extract education history"""
        education_section = self.extract_sections(text).get('Education', '')
        if not education_section:
            return []
        
        education_entries = []
        current_entry = {}
        
        # Split into lines and process
        lines = education_section.split('\n')
        for line in lines:
            # Look for degree patterns
            degree_pattern = r'(BS|MS|PhD|Bachelor|Master|Doctorate)'
            if re.search(degree_pattern, line, re.IGNORECASE):
                if current_entry:
                    education_entries.append(current_entry)
                current_entry = {'degree': line.strip()}
            
            # Look for institution
            elif re.search(r'University|College|Institute|School', line, re.IGNORECASE):
                current_entry['institution'] = line.strip()
            
            # Look for dates
            elif re.search(r'\d{4}', line):
                current_entry['dates'] = line.strip()
        
        if current_entry:
            education_entries.append(current_entry)
        
        return education_entries

    def extract_experience(self, text: str) -> List[Dict]:
        """Extract work experience"""
        experience_section = self.extract_sections(text).get('Experience', '')
        if not experience_section:
            return []
        
        experience_entries = []
        current_entry = {}
        
        # Split into lines and process
        lines = experience_section.split('\n')
        for line in lines:
            # Look for company names (usually in all caps or with special formatting)
            if re.match(r'^[A-Z\s&]+$', line.strip()):
                if current_entry:
                    experience_entries.append(current_entry)
                current_entry = {'company': line.strip()}
            
            # Look for job titles
            elif re.search(r'Engineer|Developer|Manager|Director|Lead|Architect', line, re.IGNORECASE):
                current_entry['title'] = line.strip()
            
            # Look for dates
            elif re.search(r'\d{4}', line):
                current_entry['dates'] = line.strip()
            
            # Look for bullet points
            elif line.strip().startswith(('•', '-', '*')):
                if 'bullets' not in current_entry:
                    current_entry['bullets'] = []
                current_entry['bullets'].append(line.strip())
        
        if current_entry:
            experience_entries.append(current_entry)
        
        return experience_entries

    def extract_skills(self, text: str) -> List[str]:
        """Extract skills using NER and pattern matching"""
        skills_section = self.extract_sections(text).get('Skills', '')
        if not skills_section:
            return []
        
        # Use NER to identify technical terms
        doc = self.nlp(skills_section)
        skills = []
        
        # Extract technical terms
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PRODUCT']:
                skills.append(ent.text)
        
        # Look for common skill patterns
        skill_patterns = [
            r'Python|Java|JavaScript|SQL|AWS|Docker|Kubernetes',
            r'Machine Learning|Deep Learning|Data Science',
            r'Project Management|Leadership|Communication'
        ]
        
        for pattern in skill_patterns:
            matches = re.findall(pattern, skills_section, re.IGNORECASE)
            skills.extend(matches)
        
        return list(set(skills))  # Remove duplicates

class JobParser(DocumentParser):
    def __init__(self):
        super().__init__()

    def parse(self, text: str) -> Dict[str, str]:
        """
        Parse job description text and extract structured sections
        
        Args:
            text: Raw job description text
            
        Returns:
            Dictionary with job description sections
        """
        # Create a dictionary with different parts of the job description
        sections = {
            'description': text,
            'requirements': '\n'.join(self.extract_requirements(text)),
            'responsibilities': '\n'.join(self.extract_responsibilities(text))
        }
        
        # Extract qualifications
        qualifications = self.extract_qualifications(text)
        if qualifications['required']:
            sections['required_qualifications'] = '\n'.join(qualifications['required'])
        if qualifications['preferred']:
            sections['preferred_qualifications'] = '\n'.join(qualifications['preferred'])
        
        # Extract company info
        company_info = self.extract_company_info(text)
        if company_info:
            # Get job title from the first line or beginning of text
            first_line = text.split('\n')[0].strip()
            title = first_line if len(first_line) < 100 else ""
            
            metadata = {
                'title': title,
                'company': company_info.get('name', ''),
                'location': company_info.get('location', ''),
                'industry': company_info.get('industry', 'software')  # Default to software
            }
        else:
            metadata = {'industry': 'software'}  # Default metadata
            
        return {
            'sections': sections,
            'metadata': metadata
        }

    def extract_requirements(self, text: str) -> List[str]:
        """Extract job requirements"""
        requirements = []
        
        # Common requirement section headers
        requirement_headers = [
            r'Requirements:',
            r'Job Requirements:',
            r'Required Skills:',
            r'Requirements & Skills:',
            r'What You\'ll Need:',
            r'Required:'
        ]
        
        # Find the requirements section
        for header in requirement_headers:
            if header.lower() in text.lower():
                # Split text into sections
                sections = text.split('\n')
                start_idx = -1
                
                # Find where requirements section starts
                for i, section in enumerate(sections):
                    if header.lower() in section.lower():
                        start_idx = i
                        break
                
                if start_idx != -1:
                    # Extract requirements until next section
                    for section in sections[start_idx + 1:]:
                        if section.strip() and not any(h.lower() in section.lower() for h in requirement_headers):
                            # Clean and add requirement
                            req = section.strip().strip('•-*')
                            if req:
                                requirements.append(req)
                        else:
                            break
        
        return requirements

    def extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities"""
        responsibilities = []
        
        # Common responsibility section headers
        responsibility_headers = [
            r'Responsibilities:',
            r'Job Responsibilities:',
            r'What You\'ll Do:',
            r'Role & Responsibilities:',
            r'Key Responsibilities:',
            r'Duties:'
        ]
        
        # Find the responsibilities section
        for header in responsibility_headers:
            if header.lower() in text.lower():
                # Split text into sections
                sections = text.split('\n')
                start_idx = -1
                
                # Find where responsibilities section starts
                for i, section in enumerate(sections):
                    if header.lower() in section.lower():
                        start_idx = i
                        break
                
                if start_idx != -1:
                    # Extract responsibilities until next section
                    for section in sections[start_idx + 1:]:
                        if section.strip() and not any(h.lower() in section.lower() for h in responsibility_headers):
                            # Clean and add responsibility
                            resp = section.strip().strip('•-*')
                            if resp:
                                responsibilities.append(resp)
                        else:
                            break
        
        return responsibilities

    def extract_qualifications(self, text: str) -> Dict[str, List[str]]:
        """Extract required and preferred qualifications"""
        qualifications = {
            'required': [],
            'preferred': []
        }
        
        # Common qualification section headers
        qualification_headers = [
            r'Qualifications:',
            r'Requirements & Qualifications:',
            r'Required Qualifications:',
            r'Preferred Qualifications:',
            r'What You\'ll Need:'
        ]
        
        # Find the qualifications section
        for header in qualification_headers:
            if header.lower() in text.lower():
                # Split text into sections
                sections = text.split('\n')
                start_idx = -1
                
                # Find where qualifications section starts
                for i, section in enumerate(sections):
                    if header.lower() in section.lower():
                        start_idx = i
                        break
                
                if start_idx != -1:
                    current_type = 'required'
                    # Extract qualifications until next section
                    for section in sections[start_idx + 1:]:
                        if section.strip() and not any(h.lower() in section.lower() for h in qualification_headers):
                            # Check if it's a preferred qualification
                            if 'preferred' in section.lower():
                                current_type = 'preferred'
                                continue
                            
                            # Clean and add qualification
                            qual = section.strip().strip('•-*')
                            if qual:
                                qualifications[current_type].append(qual)
                        else:
                            break
        
        return qualifications

    def extract_company_info(self, text: str) -> Dict[str, str]:
        """Extract company information"""
        company_info = {}
        
        # Use spaCy NER for company names and locations
        doc = self.nlp(text)
        
        # Extract company name
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                company_info['name'] = ent.text
                break
        
        # Extract location
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'LOC']:
                company_info['location'] = ent.text
                break
        
        # Extract industry (if mentioned)
        industry_keywords = ['industry', 'sector', 'field']
        for line in text.split('\n'):
            for keyword in industry_keywords:
                if keyword in line.lower():
                    company_info['industry'] = line.strip()
                    break
        
        # Extract company size (if mentioned)
        size_patterns = [
            r'(\d+)\+?\s*employees',
            r'(\d+)\+?\s*people',
            r'(\d+)\+?\s*staff'
        ]
        for pattern in size_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                company_info['size'] = match.group(0)
                break
        
        return company_info

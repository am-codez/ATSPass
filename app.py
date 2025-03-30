import PyPDF2
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import re
from typing import List, Dict, Tuple
import spacy
from gensim.models import Word2Vec
import numpy as np
from collections import defaultdict
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import tempfile
from functools import lru_cache

# Initialize Flask app
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_md')
    print("\nUsing spaCy model with word vectors (en_core_web_md)")
    print("This model includes word vectors for accurate similarity calculations.\n")
except OSError:
    print("Downloading spaCy model with word vectors...")
    from spacy.cli import download
    download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

# Cache for spaCy documents
@lru_cache(maxsize=100)
def get_spacy_doc(text: str) -> spacy.tokens.Doc:
    return nlp(text)

# Cache for keyword extraction
@lru_cache(maxsize=100)
def extract_keywords(text: str, use_spacy: bool = True) -> List[str]:
    """Extract keywords from text using either spaCy or NLTK."""
    if use_spacy:
        doc = get_spacy_doc(text.lower())
        keywords = [token.text for token in doc 
                   if not token.is_stop 
                   and not token.is_punct 
                   and token.is_alpha 
                   and len(token.text) > 2
                   and token.pos_ in ('NOUN', 'PROPN', 'VERB')]
    else:
        words = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in words 
                   if word.isalpha() 
                   and word not in stop_words 
                   and len(word) > 2]
    
    return list(set(keywords))

# Cache for Word2Vec model
@lru_cache(maxsize=1)
def get_word2vec_model(job_description: str) -> Word2Vec:
    sentences = [word_tokenize(sent.lower()) for sent in sent_tokenize(job_description)]
    return Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def get_semantic_synonyms(word: str, job_description: str, threshold: float = 0.6) -> List[str]:
    """Get semantically similar words using Word2Vec trained on the job description."""
    model = get_word2vec_model(job_description)
    try:
        similar_words = [w for w, score in model.wv.most_similar(word, topn=5) if score > threshold]
        return similar_words
    except KeyError:
        return []

def calculate_keyword_density(text: str, keyword: str) -> float:
    """Calculate the density of a keyword in the text."""
    words = word_tokenize(text.lower())
    total_words = len(words)
    if total_words == 0:
        return 0.0
    keyword_count = sum(1 for word in words if word == keyword.lower())
    return (keyword_count / total_words) * 100

def optimize_section(section_text: str, missing_keywords: List[str], job_description: str) -> str:
    """Optimize a resume section by intelligently incorporating missing keywords."""
    doc = get_spacy_doc(section_text)
    optimized_sentences = []
    
    # Pre-calculate similarities for all keywords and their semantic variations
    keyword_similarities = {}
    for keyword in missing_keywords:
        keyword_doc = get_spacy_doc(keyword)
        # Get semantic variations of the keyword
        semantic_variations = get_semantic_synonyms(keyword, job_description, threshold=0.5)
        all_keywords = [keyword] + semantic_variations
        
        for sent in doc.sents:
            sent_text = sent.text
            sent_doc = get_spacy_doc(sent_text)
            
            # Calculate similarity with all variations
            max_similarity = 0
            best_variation = keyword
            for variation in all_keywords:
                variation_doc = get_spacy_doc(variation)
                similarity = sent_doc.similarity(variation_doc)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_variation = variation
            
            if max_similarity > 0.3:
                if sent_text not in keyword_similarities:
                    keyword_similarities[sent_text] = []
                keyword_similarities[sent_text].append((best_variation, max_similarity))
    
    for sent in doc.sents:
        sent_text = sent.text
        sent_doc = get_spacy_doc(sent_text)
        
        # Check for action verbs that can be enhanced
        for token in sent_doc:
            if token.pos_ == 'VERB' and token.text.lower() in ['did', 'made', 'worked', 'helped', 'assisted', 'supported']:
                for keyword in missing_keywords:
                    keyword_doc = get_spacy_doc(keyword)
                    if any(t.pos_ == 'VERB' for t in keyword_doc):
                        sent_text = sent_text.replace(token.text, keyword)
                        break
        
        # Add the most relevant keywords if they fit naturally
        if sent_text in keyword_similarities:
            relevant_keywords = sorted(keyword_similarities[sent_text], key=lambda x: x[1], reverse=True)
            
            # Add up to 2 most relevant keywords if they fit
            for keyword, similarity in relevant_keywords[:2]:
                current_density = calculate_keyword_density(sent_text, keyword)
                if keyword.lower() not in sent_text.lower() and current_density < 2.0:
                    if sent_text.endswith('.'):
                        sent_text = sent_text[:-1] + f", {keyword}."
                    else:
                        sent_text += f" {keyword}"
        
        optimized_sentences.append(sent_text)
    
    return ' '.join(optimized_sentences)

def format_resume_section(section_name: str, content: List[str]) -> str:
    """Format a resume section with proper spacing and structure."""
    if not content:
        return ""
    
    # Format section header
    formatted = f"\n{section_name}\n"
    
    # Add content with proper indentation
    for line in content:
        if line.strip():
            # If line starts with a bullet point or is a job title, don't indent
            if line.strip().startswith(('•', '-', '*')) or any(line.strip().startswith(title) for title in ['DevOps Engineer', 'Site Reliability Engineer', 'Cloud Engineer', 'Software Engineer']):
                formatted += f"\n{line.strip()}"
            else:
                formatted += f"\n    {line.strip()}"
    
    return formatted

def enhance_resume(original_resume: str, missing_keywords: List[str], job_description: str) -> str:
    """Enhance the resume by strategically incorporating missing keywords."""
    # Split resume into sections
    sections = {
        'contact': [],
        'summary': [],
        'skills': [],
        'experience': [],
        'education': [],
        'projects': [],
        'certifications': [],
        'additional': []
    }
    
    # Enhanced section detection
    lines = original_resume.split('\n')
    current_section = None
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Contact information detection
        if '@' in line or 'linkedin.com' in line or 'github.com' in line or any(c.isdigit() for c in line):
            current_section = 'contact'
        # Section headers detection
        elif any(word in line_lower for word in ['experience', 'work experience', 'employment']):
            current_section = 'experience'
        elif any(word in line_lower for word in ['education', 'academic', 'qualification']):
            current_section = 'education'
        elif any(word in line_lower for word in ['skills', 'technical skills', 'expertise']):
            current_section = 'skills'
        elif any(word in line_lower for word in ['summary', 'objective', 'profile', 'about']):
            current_section = 'summary'
        elif any(word in line_lower for word in ['projects', 'portfolio', 'achievements']):
            current_section = 'projects'
        elif any(word in line_lower for word in ['certifications', 'certificates']):
            current_section = 'certifications'
        elif any(word in line_lower for word in ['additional', 'other']):
            current_section = 'additional'
        
        if current_section and line.strip():
            sections[current_section].append(line)
    
    # Optimize each section with section-specific strategies
    optimized_sections = {}
    for section_name, section_lines in sections.items():
        section_text = '\n'.join(section_lines)
        if section_name == 'summary':
            # Add more keywords to summary section
            optimized_text = optimize_section(section_text, missing_keywords, job_description)
            # Add a few more relevant keywords if space allows
            if len(optimized_text.split()) < 100:  # If summary is not too long
                for keyword in missing_keywords[:2]:
                    if keyword.lower() not in optimized_text.lower():
                        optimized_text += f" {keyword}"
        elif section_name == 'contact':
            # Keep contact information as is
            optimized_text = section_text
        else:
            optimized_text = optimize_section(section_text, missing_keywords, job_description)
        optimized_sections[section_name] = optimized_text.split('\n')
    
    # Format and combine sections in the correct order
    formatted_resume = []
    
    # Contact information (if exists)
    if optimized_sections['contact']:
        formatted_resume.append('\n'.join(optimized_sections['contact']))
        formatted_resume.append('')  # Add blank line after contact
    
    # Summary
    if optimized_sections['summary']:
        formatted_resume.append(format_resume_section('Professional Summary', optimized_sections['summary']))
        formatted_resume.append('')  # Add blank line after summary
    
    # Skills
    if optimized_sections['skills']:
        formatted_resume.append(format_resume_section('Technical Skills', optimized_sections['skills']))
        formatted_resume.append('')  # Add blank line after skills
    
    # Experience
    if optimized_sections['experience']:
        formatted_resume.append(format_resume_section('Professional Experience', optimized_sections['experience']))
        formatted_resume.append('')  # Add blank line after experience
    
    # Projects
    if optimized_sections['projects']:
        formatted_resume.append(format_resume_section('Projects', optimized_sections['projects']))
        formatted_resume.append('')  # Add blank line after projects
    
    # Certifications
    if optimized_sections['certifications']:
        formatted_resume.append(format_resume_section('Certifications', optimized_sections['certifications']))
        formatted_resume.append('')  # Add blank line after certifications
    
    # Education
    if optimized_sections['education']:
        formatted_resume.append(format_resume_section('Education', optimized_sections['education']))
        formatted_resume.append('')  # Add blank line after education
    
    # Additional Information
    if optimized_sections['additional']:
        formatted_resume.append(format_resume_section('Additional Information', optimized_sections['additional']))
    
    return '\n'.join(formatted_resume)

def analyze_and_modify_resume(resume_path: str, jd_path: str) -> Dict:
    """Analyze resume against job description and suggest improvements."""
    try:
        # Read the text directly from the files
        with open(resume_path, 'r', encoding='utf-8') as f:
            resume_text = f.read()
        with open(jd_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
        
        # Extract keywords using spaCy for better accuracy
        resume_keywords = extract_keywords(resume_text, use_spacy=True)
        jd_keywords = extract_keywords(jd_text, use_spacy=True)
        
        # Calculate initial match score
        initial_score = calculate_match_score(resume_keywords, jd_keywords)
        
        # Find missing keywords
        missing_keywords = list(set(jd_keywords) - set(resume_keywords))
        
        # Enhance resume with semantic analysis
        enhanced_resume = enhance_resume(resume_text, missing_keywords, jd_text)
        
        # Extract keywords from enhanced resume
        enhanced_keywords = extract_keywords(enhanced_resume, use_spacy=True)
        
        # Calculate new match score
        enhanced_score = calculate_match_score(enhanced_keywords, jd_keywords)
        
        # Find added keywords
        added_keywords = list(set(enhanced_keywords) - set(resume_keywords))
        
        return {
            'match_score': round(initial_score, 2),
            'missing_keywords': missing_keywords,
            'enhanced_score': round(enhanced_score, 2),
            'added_keywords': added_keywords,
            'modified_resume': enhanced_resume
        }
        
    except Exception as e:
        print(f"Error in analyze_and_modify_resume: {str(e)}")
        return None

def calculate_match_score(resume_keywords: List[str], jd_keywords: List[str]) -> float:
    """Calculate the match score between resume and job description keywords with semantic similarity."""
    if not jd_keywords:
        return 0.0
    
    # Exact matches (weighted higher)
    exact_matches = set(resume_keywords) & set(jd_keywords)
    
    # Semantic matches with higher threshold
    semantic_matches = 0
    jd_vectors = [get_spacy_doc(keyword).vector for keyword in jd_keywords if len(get_spacy_doc(keyword).vector) > 0]
    
    if jd_vectors:
        avg_jd_vector = np.mean(jd_vectors, axis=0)
        
        for r_keyword in resume_keywords:
            r_vector = get_spacy_doc(r_keyword).vector
            if len(r_vector) > 0:
                similarity = np.dot(avg_jd_vector, r_vector) / (np.linalg.norm(avg_jd_vector) * np.linalg.norm(r_vector))
                if similarity > 0.8:  # Increased threshold for better accuracy
                    semantic_matches += 1
    
    # Weight exact matches more heavily
    total_matches = (len(exact_matches) * 1.5) + semantic_matches
    return (total_matches / len(jd_keywords)) * 100

# Flask Routes
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    try:
        # Get text data from the request
        resume_text = request.form.get('resume_text')
        job_description = request.form.get('job_description')
        
        if not resume_text or not job_description:
            return jsonify({'error': 'Both resume text and job description are required'}), 400
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as resume_temp:
            resume_temp.write(resume_text)
            resume_path = resume_temp.name
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as jd_temp:
            jd_temp.write(job_description)
            jd_path = jd_temp.name
        
        # Analyze the resume
        result = analyze_and_modify_resume(resume_path, jd_path)
        
        # Clean up temporary files
        os.unlink(resume_path)
        os.unlink(jd_path)
        
        if result:
            return jsonify({
                'match_score': result.get('match_score', 0),
                'missing_keywords': result.get('missing_keywords', []),
                'enhanced_score': result.get('enhanced_score', 0),
                'added_keywords': result.get('added_keywords', []),
                'enhanced_resume': result.get('modified_resume', '')
            })
        else:
            return jsonify({'error': 'Failed to analyze resume'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
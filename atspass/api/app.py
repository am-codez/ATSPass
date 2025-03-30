"""
Flask API for resume analysis and optimization.
"""

import os
import tempfile
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from ..resume_analyzer.analyzer import ResumeAnalyzer

# Initialize Flask app
app = Flask(__name__, static_folder='../static')
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize resume analyzer
analyzer = ResumeAnalyzer()

@app.route('/')
def index():
    return send_from_directory('../static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('../static', filename)

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
        result = analyzer.analyze_and_modify_resume(resume_text, job_description)
        
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
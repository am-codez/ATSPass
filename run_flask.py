from flask import Flask, send_from_directory, request, jsonify
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import the actual implementation
from src.app import analyze_resume_job_match, enhance_resume

app = Flask(__name__, static_folder="static")

# API endpoints using the real implementation
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    try:
        # Get form data
        resume_text = request.form.get('resume_text', '')
        job_description = request.form.get('job_description', '')
        
        if not resume_text or not job_description:
            return jsonify({'error': 'Both resume and job description are required'}), 400
        
        # Call the actual implementation instead of returning dummy data
        results = analyze_resume_job_match(resume_text, job_description)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/enhance', methods=['POST'])
def api_enhance():
    try:
        # Get form data
        resume_text = request.form.get('resume_text', '')
        job_description = request.form.get('job_description', '')
        
        if not resume_text or not job_description:
            return jsonify({'error': 'Both resume and job description are required'}), 400
        
        # Call the actual implementation instead of returning dummy data
        results = enhance_resume(resume_text, job_description)
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory("static", path)

if __name__ == "__main__":
    print("Starting Flask server on http://127.0.0.1:8080")
    app.run(debug=True, port=8080, host='127.0.0.1')

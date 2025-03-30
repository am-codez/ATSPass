from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import traceback
from resume_parser import analyze_and_modify_resume
import tempfile

app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Serve the main page
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Serve static files
@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    try:
        print("API request received for /api/analyze")
        
        # Get text data from the request
        resume_text = request.form.get('resume_text')
        job_description = request.form.get('job_description')
        
        print(f"Resume text length: {len(resume_text) if resume_text else 0}")
        print(f"Job description length: {len(job_description) if job_description else 0}")
        
        if not resume_text or not job_description:
            print("Error: Missing resume text or job description")
            return jsonify({'error': 'Both resume text and job description are required'}), 400
        
        # Create temporary files with the text content
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as resume_temp:
            resume_temp.write(resume_text)
            resume_temp.flush()
            resume_path = resume_temp.name
            print(f"Resume saved to temporary file: {resume_path}")
        
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as jd_temp:
            jd_temp.write(job_description)
            jd_temp.flush()
            jd_path = jd_temp.name
            print(f"Job description saved to temporary file: {jd_path}")
        
        # Analyze the resume
        print("Calling analyze_and_modify_resume function")
        result = analyze_and_modify_resume(resume_path, jd_path)
        
        # Clean up temporary files
        os.unlink(resume_path)
        os.unlink(jd_path)
        print("Temporary files cleaned up")
        
        if result:
            print("Analysis completed successfully")
            response_data = {
                'match_score': result.get('match_score', 0),
                'missing_keywords': result.get('missing_keywords', []),
                'enhanced_score': result.get('enhanced_score', 0),
                'added_keywords': result.get('added_keywords', []),
                'enhanced_resume': result.get('modified_resume', '')
            }
            print(f"Response data prepared: match_score={response_data['match_score']}, missing_keywords_count={len(response_data['missing_keywords'])}")
            return jsonify(response_data)
        else:
            print("Error: analyze_and_modify_resume returned None")
            return jsonify({'error': 'Failed to analyze resume'}), 500
            
    except Exception as e:
        print(f"Exception in analyze_resume: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download', methods=['POST'])
def download_analysis():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Create a temporary file for the analysis
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8') as temp_file:
            # Write the analysis to the file
            temp_file.write("Resume Analysis and Modification Results\n")
            temp_file.write("-" * 50 + "\n\n")
            
            temp_file.write("Keywords from Job Description:\n")
            if 'keywords' in data and isinstance(data['keywords'], dict):
                for keyword, synonyms in data['keywords'].items():
                    temp_file.write(f"\n{keyword}:\n")
                    for synonym in synonyms:
                        temp_file.write(f"- {synonym}\n")
            
            temp_file.write("\nOriginal Resume:\n")
            temp_file.write("-" * 50 + "\n")
            temp_file.write(data.get('originalResume', ''))
            
            temp_file.write("\n\nModified Resume:\n")
            temp_file.write("-" * 50 + "\n")
            temp_file.write(data.get('modifiedResume', ''))
            
            temp_file.write("\n\nJob Description:\n")
            temp_file.write("-" * 50 + "\n")
            temp_file.write(data.get('jobDescription', ''))
            
            temp_file_path = temp_file.name
        
        # Send the file
        return send_file(
            temp_file_path,
            as_attachment=True,
            download_name='resume_analysis.txt',
            mimetype='text/plain'
        )
        
    except Exception as e:
        print(f"Exception in download_analysis: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001) 
# ATSPass - Advanced Resume Optimization Tool

ATSPass is an intelligent resume optimization tool that helps job seekers improve their resumes by analyzing job descriptions and suggesting improvements. It uses advanced NLP techniques to identify missing keywords and optimize content while maintaining readability.

## Features

- **Smart Keyword Analysis**: Identifies important keywords from job descriptions
- **Semantic Matching**: Uses spaCy and Word2Vec for accurate semantic similarity calculations
- **Intelligent Optimization**: Suggests improvements while maintaining natural language
- **Section-Aware Processing**: Understands and preserves resume structure
- **Format Preservation**: Maintains professional resume formatting
- **Real-time Analysis**: Provides instant feedback and optimization suggestions


1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the spaCy model:
```bash
python -m spacy download en_core_web_md
```


## Usage

1. Start the application:
```bash
python -m atspass
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter your resume text and job description in the web interface

4. Click "Analyze" to get optimization suggestions



## Acknowledgments

- spaCy team for the excellent NLP library
- NLTK team for the comprehensive NLP toolkit
- Gensim team for the Word2Vec implementation
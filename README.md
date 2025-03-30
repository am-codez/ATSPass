# Resume Optimization System

A simple NLP-based system that analyzes resumes against job descriptions to provide optimization suggestions.

## Features

- PDF and DOCX resume parsing
- Job description analysis
- Skills matching
- Improvement recommendations

## Project Structure

```
NLP_ATS/
├── src/
│   ├── parser.py      # Document parsing
│   ├── matcher.py     # Resume-job matching
│   └── constants.py   # Predefined values
├── app.py            # Streamlit interface
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download spaCy model:
```bash
python -m spacy download en_core_web_sm
```

4. Run the application:
```bash
streamlit run app.py
```

## Usage

1. Upload your resume (PDF or DOCX)
2. Paste the job description
3. Get analysis results:
   - Match score
   - Skills comparison
   - Improvement suggestions

## Technologies Used

- Python 3.8+
- spaCy for NLP
- scikit-learn for text matching
- Streamlit for UI

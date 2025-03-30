import PyPDF2
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import re
import openai
from typing import List, Dict, Tuple

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('wordnet')

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        # Check if file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        # Open the PDF file in binary mode
        with open(pdf_path, 'rb') as file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get the number of pages
            num_pages = len(pdf_reader.pages)
            
            # Extract text from all pages
            text = ""
            for page_num in range(num_pages):
                # Get the page object
                page = pdf_reader.pages[page_num]
                # Extract text from the page
                text += page.extract_text()
                
            return text
            
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

def get_synonyms(word):
    """
    Get synonyms for a given word using WordNet.
    
    Args:
        word (str): Word to find synonyms for
        
    Returns:
        list: List of synonyms
    """
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
    return list(set(synonyms))  # Remove duplicates

def extract_keywords(text: str) -> List[str]:
    """Extract keywords from text, filtering out stopwords and non-alphabetic words."""
    # Tokenize the text
    words = word_tokenize(text.lower())
    
    # Get stopwords
    stop_words = set(stopwords.words('english'))
    
    # Filter out stopwords and non-alphabetic words
    keywords = [word for word in words 
                if word.isalpha() 
                and word not in stop_words 
                and len(word) > 2]
    
    return list(set(keywords))  # Remove duplicates

def calculate_match_score(resume_keywords: List[str], jd_keywords: List[str]) -> float:
    """Calculate the match score between resume and job description keywords."""
    if not jd_keywords:
        return 0.0
    
    matched_keywords = set(resume_keywords) & set(jd_keywords)
    return (len(matched_keywords) / len(jd_keywords)) * 100

def enhance_resume_with_chatgpt(original_resume: str, missing_keywords: List[str]) -> str:
    """Use ChatGPT to enhance the resume with synonyms for missing keywords."""
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv(''))
        
        # Prepare the prompt
        prompt = f"""Please enhance this resume by incorporating these keywords naturally: {', '.join(missing_keywords)}.
        Rules:
        1. Keep the tone professional
        2. Don't add any hard skills that weren't in the original resume
        3. Don't increase word count by more than 30 words
        4. Only replace existing words with synonyms
        5. Maintain the original structure and format
        
        Original resume:
        {original_resume}
        
        Enhanced resume:"""
        
        # Make API call to ChatGPT
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a professional resume writer who helps optimize resumes for job descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error in ChatGPT API call: {str(e)}")
        return original_resume

def analyze_and_modify_resume(resume_path: str, jd_path: str) -> Dict:
    """Analyze resume against job description and suggest improvements."""
    try:
        # Read the text directly from the files
        with open(resume_path, 'r', encoding='utf-8') as f:
            resume_text = f.read()
        with open(jd_path, 'r', encoding='utf-8') as f:
            jd_text = f.read()
        
        # Extract keywords
        resume_keywords = extract_keywords(resume_text)
        jd_keywords = extract_keywords(jd_text)
        
        # Calculate initial match score
        initial_score = calculate_match_score(resume_keywords, jd_keywords)
        
        # Find missing keywords
        missing_keywords = list(set(jd_keywords) - set(resume_keywords))
        
        # Enhance resume with ChatGPT
        enhanced_resume = enhance_resume_with_chatgpt(resume_text, missing_keywords)
        
        # Extract keywords from enhanced resume
        enhanced_keywords = extract_keywords(enhanced_resume)
        
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

def main():
    # File paths
    resume_path = "test_resume.pdf"  # Your resume PDF
    jd_path = "description.txt"      # Your job description text file
    
    # Analyze and modify
    result = analyze_and_modify_resume(resume_path, jd_path)
    
    if result:
        print("\nResume Analysis and Modification Results:")
        print("-" * 50)
        
        print("\nKeywords from Job Description:")
        for keyword in result['missing_keywords']:
            print(f"- {keyword}")
        
        print("\nOriginal Resume (First 500 characters):")
        print("-" * 50)
        print(result['modified_resume'][:500] + "...")
        
        # Save results to a file
        output_file = "resume_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Resume Analysis and Modification Results\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("Keywords from Job Description:\n")
            for keyword in result['missing_keywords']:
                f.write(f"- {keyword}\n")
            
            f.write("\nOriginal Resume:\n")
            f.write("-" * 50 + "\n")
            f.write(result['modified_resume'])
        
        print(f"\nDetailed analysis has been saved to: {output_file}")

if __name__ == "__main__":
    main() 
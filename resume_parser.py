import PyPDF2
import os
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('stopwords')
    nltk.download('wordnet')

def extract_text_from_pdf(pdf_path):
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

def extract_keywords_from_jd(jd_text):
    """
    Extract important keywords from job description.
    
    Args:
        jd_text (str): Job description text
        
    Returns:
        dict: Dictionary of keywords and their synonyms
    """
    # Tokenize the text
    words = word_tokenize(jd_text.lower())
    
    # Remove stopwords and non-alphabetic characters
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    
    # Get part of speech tags
    tagged = pos_tag(words)
    
    # Focus on nouns, verbs, and adjectives
    important_words = []
    for word, tag in tagged:
        if tag.startswith(('NN', 'VB', 'JJ')):  # Nouns, Verbs, Adjectives
            important_words.append(word)
    
    # Get synonyms for important words
    keywords = {}
    for word in important_words:
        synonyms = get_synonyms(word)
        if synonyms:
            keywords[word] = synonyms
    
    return keywords

def replace_keywords_in_text(text, keywords):
    """
    Replace words in text with their synonyms from keywords.
    
    Args:
        text (str): Original text
        keywords (dict): Dictionary of keywords and their synonyms
        
    Returns:
        str: Modified text with replacements
    """
    # Split text into sentences
    sentences = sent_tokenize(text)
    modified_sentences = []
    
    for sentence in sentences:
        words = word_tokenize(sentence)
        modified_words = []
        
        for word in words:
            word_lower = word.lower()
            # Check if word is a keyword
            if word_lower in keywords:
                # Randomly choose a synonym
                import random
                synonym = random.choice(keywords[word_lower])
                # Preserve original capitalization
                if word[0].isupper():
                    synonym = synonym.capitalize()
                modified_words.append(synonym)
            else:
                modified_words.append(word)
        
        modified_sentences.append(' '.join(modified_words))
    
    return ' '.join(modified_sentences)

def read_text_file(file_path):
    """
    Read text from a text file.
    
    Args:
        file_path (str): Path to the text file
        
    Returns:
        str: Content of the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                return file.read()
        except Exception as e:
            print(f"Error reading text file with latin-1 encoding: {str(e)}")
            return None
    except Exception as e:
        print(f"Error reading text file: {str(e)}")
        return None

def analyze_and_modify_resume(resume_path, jd_path):
    """
    Analyze resume and job description, then modify resume with relevant synonyms.
    
    Args:
        resume_path (str): Path to resume file
        jd_path (str): Path to job description file
        
    Returns:
        dict: Dictionary containing analysis results and modified content
    """
    # Extract text from resume
    resume_text = read_text_file(resume_path)
    if not resume_text:
        return None
    
    # Read job description
    jd_text = read_text_file(jd_path)
    if not jd_text:
        return None
    
    # Extract keywords from job description
    keywords = extract_keywords_from_jd(jd_text)
    
    # Calculate initial match score
    resume_words = set(word.lower() for word in word_tokenize(resume_text) if word.isalpha())
    jd_words = set(word.lower() for word in word_tokenize(jd_text) if word.isalpha())
    matching_words = resume_words.intersection(jd_words)
    match_score = int((len(matching_words) / len(jd_words)) * 100)
    
    # Find missing keywords
    missing_keywords = [word for word in keywords.keys() if word not in resume_words]
    
    # Replace keywords in resume
    modified_resume = replace_keywords_in_text(resume_text, keywords)
    
    # Calculate enhanced match score
    modified_words = set(word.lower() for word in word_tokenize(modified_resume) if word.isalpha())
    enhanced_matching = modified_words.intersection(jd_words)
    enhanced_score = int((len(enhanced_matching) / len(jd_words)) * 100)
    
    # Find added keywords
    added_keywords = [word for word in modified_words if word in jd_words and word not in resume_words]
    
    return {
        'match_score': match_score,
        'missing_keywords': missing_keywords,
        'enhanced_score': enhanced_score,
        'added_keywords': added_keywords,
        'modified_resume': modified_resume,
        'original_resume': resume_text,
        'job_description': jd_text
    }

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
        for keyword, synonyms in result['keywords'].items():
            print(f"\n{keyword}:")
            for synonym in synonyms:
                print(f"- {synonym}")
        
        print("\nOriginal Resume (First 500 characters):")
        print("-" * 50)
        print(result['original_resume'][:500] + "...")
        
        print("\nModified Resume (First 500 characters):")
        print("-" * 50)
        print(result['modified_resume'][:500] + "...")
        
        # Save results to a file
        output_file = "resume_analysis.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Resume Analysis and Modification Results\n")
            f.write("-" * 50 + "\n\n")
            
            f.write("Keywords from Job Description:\n")
            for keyword, synonyms in result['keywords'].items():
                f.write(f"\n{keyword}:\n")
                for synonym in synonyms:
                    f.write(f"- {synonym}\n")
            
            f.write("\nOriginal Resume:\n")
            f.write("-" * 50 + "\n")
            f.write(result['original_resume'])
            
            f.write("\n\nModified Resume:\n")
            f.write("-" * 50 + "\n")
            f.write(result['modified_resume'])
            
            f.write("\n\nJob Description:\n")
            f.write("-" * 50 + "\n")
            f.write(result['job_description'])
        
        print(f"\nDetailed analysis has been saved to: {output_file}")

if __name__ == "__main__":
    main() 
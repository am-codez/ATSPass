"""
NLP utilities for resume analysis and optimization.
"""

import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from gensim.models import Word2Vec
from typing import List
from functools import lru_cache

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

@lru_cache(maxsize=100)
def get_spacy_doc(text: str) -> spacy.tokens.Doc:
    """Get a cached spaCy document for the given text."""
    return nlp(text)

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

@lru_cache(maxsize=1)
def get_word2vec_model(job_description: str) -> Word2Vec:
    """Get a cached Word2Vec model trained on the job description."""
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
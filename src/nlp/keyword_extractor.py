from src.preprocessing.cleaner import TextCleaner
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List, Dict, Tuple
import re

'''`keyword_extractor.py`:
     - Custom TF-IDF implementation
     - Industry-specific keyword weighting
     - Context-aware extraction'''

class KeywordExtractor:
    def __init__(self, industry: str):
        """
        Initialize the KeywordExtractor with industry-specific settings
        
        Args:
            industry: The industry to focus on for keyword extraction
        """
        # Store the industry for later use
        self.industry = industry
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.cleaner = TextCleaner()
        
        # Initialize a basic TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,  # Limit to 1000 features to avoid noise
            ngram_range=(1, 2),  # Extract both single words and bigrams
            stop_words='english'  # Use English stopwords list from sklearn
        )
        
        # Load industry-specific keyword weights
        self.industry_weights = self._load_industry_weights()

    def _load_industry_weights(self) -> Dict[str, float]:
        """
        Load industry-specific keyword weights
        
        Returns:
            Dictionary mapping keywords to their importance weights for the specific industry
        """
        # Software Engineering industry
        if self.industry.lower() in ['software', 'software engineering', 'tech']:
            return {
                # Programming languages
                'python': 1.5, 'java': 1.5, 'javascript': 1.5, 'typescript': 1.4,
                'c++': 1.4, 'c#': 1.4, 'go': 1.3, 'rust': 1.3, 'php': 1.2,
                
                # Web frameworks
                'react': 1.4, 'angular': 1.4, 'vue': 1.4, 'django': 1.3,
                'flask': 1.3, 'spring': 1.3, 'node': 1.3, 'express': 1.3,
                
                # Cloud & DevOps
                'aws': 1.4, 'azure': 1.3, 'gcp': 1.3, 'docker': 1.4,
                'kubernetes': 1.4, 'jenkins': 1.2, 'terraform': 1.3,
                
                # Databases
                'sql': 1.3, 'mongodb': 1.3, 'postgresql': 1.3,
                'mysql': 1.2, 'redis': 1.2, 'elasticsearch': 1.3,
            }
        
        # Data Science industry
        elif self.industry.lower() in ['data science', 'machine learning', 'ai']:
            return {
                # Languages & tools
                'python': 1.5, 'r': 1.4, 'sql': 1.3, 
                'pandas': 1.5, 'numpy': 1.4, 'scikit-learn': 1.5,
                
                # ML frameworks
                'tensorflow': 1.5, 'pytorch': 1.5, 'keras': 1.4,
                
                # ML concepts
                'machine learning': 1.5, 'deep learning': 1.5, 'nlp': 1.4,
                'neural networks': 1.4, 'feature engineering': 1.3
            }
        
        # Default case - return empty dictionary
        return {}
        
    def _preprocess_text(self, text: str) -> str:
        """Simple text preprocessing for keyword extraction"""
        # Clean the text
        cleaned_text = self.cleaner.clean_text(text)
        
        # Tokenize
        tokens = self.cleaner.tokenize_text(cleaned_text)
        
        # Remove stopwords and lemmatize
        tokens = self.cleaner.remove_stopwords(tokens)
        tokens = self.cleaner.lemmatize_tokens(tokens)
        
        # Join back into a string
        return ' '.join(tokens)
        
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF and industry-specific weighting
        
        Args:
            text: The text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples, sorted by score
        """
        # Handle empty text
        if not text or text.strip() == '':
            return []
            
        # Preprocess the text
        processed_text = self._preprocess_text(text)
        if not processed_text:
            return []
            
        # Use TF-IDF to extract keywords
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([processed_text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.toarray()[0]
        
        # Apply industry-specific weights
        weighted_keywords = []
        for word, base_score in zip(feature_names, tfidf_scores):
            if base_score == 0:
                continue
                
            # Apply industry weight if available
            industry_weight = self.industry_weights.get(word, 1.0)
            weighted_score = base_score * industry_weight
            
            weighted_keywords.append((word, weighted_score))
        
        # Sort by weight and return top N
        weighted_keywords.sort(key=lambda x: x[1], reverse=True)
        return weighted_keywords[:top_n]
        
    def extract_context_aware_keywords(self, text: str, context: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords with awareness of document context
        
        This boosts keywords that appear in both the main text and the context.
        
        Args:
            text: The main text to extract keywords from
            context: The context to consider for relevance
            top_n: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples, sorted by relevance score
        """
        if not text or not context:
            return self.extract_keywords(text, top_n)
            
        # Extract keywords from both text and context
        main_keywords = self.extract_keywords(text, top_n * 2)
        context_keywords = self.extract_keywords(context, top_n * 2)
        
        # Combine and boost keywords that appear in both
        combined_scores = {}
        
        # Add all keywords from main text
        for word, score in main_keywords:
            combined_scores[word] = score
            
        # Boost scores for keywords that also appear in context
        for word, context_score in context_keywords:
            if word in combined_scores:
                # Boost words that appear in both text and context
                combined_scores[word] *= 1.5
                
        # Sort by score and return top N
        result = [(word, score) for word, score in combined_scores.items()]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result[:top_n]
   
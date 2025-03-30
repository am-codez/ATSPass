"""
Similarity Matching Module
Implements multiple similarity metrics for resume-job matching.
"""

from typing import Dict, List, Set, Tuple, Optional, Union, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Import from other modules
from src.nlp.semantic_analyzer import SemanticAnalyzer

class SimilarityMatcher:
    """
    Calculate various similarity metrics between documents
    - Jaccard similarity
    - Cosine similarity 
    - Semantic similarity
    - Multi-metric combination
    """
    
    def __init__(self, semantic_analyzer: Optional[SemanticAnalyzer] = None):
        """
        Initialize the similarity calculator
        
        Args:
            semantic_analyzer: Optional SemanticAnalyzer object for semantic similarity
        """
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer()
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),  # Include single words and bigrams
            max_features=5000
        )
        
        # Default weights for multi-metric combination
        self.default_weights = {
            'jaccard': 0.2,
            'cosine': 0.3,
            'semantic': 0.5
        }
    
    def jaccard_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate Jaccard similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Jaccard similarity score between 0 and 1
        """
        # Tokenize texts (convert to lowercase and split by non-alphanumeric characters)
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        if union == 0:  # Prevent division by zero
            return 0.0
            
        return intersection / union
    
    def cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts using TF-IDF vectors
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        # Convert texts to TF-IDF vectors
        tfidf_matrix = self.tfidf_vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        dot_product = (tfidf_matrix[0] * tfidf_matrix[1].T).toarray()[0, 0]
        norm1 = np.sqrt((tfidf_matrix[0] * tfidf_matrix[0].T).toarray()[0, 0])
        norm2 = np.sqrt((tfidf_matrix[1] * tfidf_matrix[1].T).toarray()[0, 0])
        
        if norm1 == 0 or norm2 == 0:  # Prevent division by zero
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using BERT embeddings
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity score between 0 and 1
        """
        return self.semantic_analyzer.calculate_similarity(text1, text2)
    
    def multi_metric_similarity(self, 
                               text1: str, 
                               text2: str, 
                               weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate combined similarity score using multiple metrics
        
        Args:
            text1: First text
            text2: Second text
            weights: Optional dictionary with weights for each metric
                     (defaults to self.default_weights if not provided)
            
        Returns:
            Dictionary with individual and combined similarity scores
        """
        # Use default weights if none provided
        if weights is None:
            weights = self.default_weights
        
        # Calculate individual similarities
        jaccard = self.jaccard_similarity(text1, text2)
        cosine = self.cosine_similarity(text1, text2)
        semantic = self.semantic_similarity(text1, text2)
        
        # Calculate weighted combination
        combined_score = (
            jaccard * weights.get('jaccard', 0.2) +
            cosine * weights.get('cosine', 0.3) +
            semantic * weights.get('semantic', 0.5)
        )
        
        # Return all scores
        return {
            'jaccard': jaccard,
            'cosine': cosine,
            'semantic': semantic,
            'combined': combined_score
        }
    
    def calculate_section_similarities(self, 
                                      doc1_sections: Dict[str, str], 
                                      doc2_sections: Dict[str, str],
                                      method: str = "multi") -> Dict[str, Dict[str, float]]:
        """
        Calculate similarities between corresponding sections of two documents
        
        Args:
            doc1_sections: Dictionary mapping section names to text for first document
            doc2_sections: Dictionary mapping section names to text for second document
            method: Similarity method to use ('jaccard', 'cosine', 'semantic', or 'multi')
            
        Returns:
            Dictionary mapping section names to similarity scores
        """
        similarities = {}
        
        # Find common sections
        common_sections = set(doc1_sections.keys()) & set(doc2_sections.keys())
        
        # Calculate similarity for each common section
        for section in common_sections:
            text1 = doc1_sections[section]
            text2 = doc2_sections[section]
            
            if not text1 or not text2:  # Skip if either section is empty
                similarities[section] = {'combined': 0.0} if method == 'multi' else 0.0
                continue
            
            # Calculate appropriate similarity based on method
            if method == 'jaccard':
                similarities[section] = self.jaccard_similarity(text1, text2)
            elif method == 'cosine':
                similarities[section] = self.cosine_similarity(text1, text2)
            elif method == 'semantic':
                similarities[section] = self.semantic_similarity(text1, text2)
            else:  # multi-metric
                similarities[section] = self.multi_metric_similarity(text1, text2)
                
        return similarities
    
    def calculate_overall_similarity(self, 
                                   doc1_sections: Dict[str, str], 
                                   doc2_sections: Dict[str, str],
                                   section_weights: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate overall similarity between two documents with section weighting
        
        Args:
            doc1_sections: Dictionary mapping section names to text for first document
            doc2_sections: Dictionary mapping section names to text for second document
            section_weights: Optional dictionary with weights for each section
            
        Returns:
            Overall similarity score between 0 and 1
        """
        # Get section similarities using multi-metric approach
        section_similarities = self.calculate_section_similarities(
            doc1_sections, 
            doc2_sections, 
            method='multi'
        )
        
        # Default to equal weighting if no weights provided
        if section_weights is None:
            common_sections = set(doc1_sections.keys()) & set(doc2_sections.keys())
            if not common_sections:
                return 0.0
            
            section_weights = {section: 1.0 / len(common_sections) for section in common_sections}
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        
        for section, similarity in section_similarities.items():
            if section in section_weights:
                weight = section_weights[section]
                # Get the combined score if it's a dictionary
                score = similarity['combined'] if isinstance(similarity, dict) else similarity
                
                weighted_sum += score * weight
                total_weight += weight
        
        # Return weighted average or 0 if no weights
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def get_document_similarities(self, 
                                 doc1: Dict[str, Any], 
                                 doc2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get comprehensive similarity analysis between two documents
        
        Args:
            doc1: First document with sections and metadata
            doc2: Second document with sections and metadata
            
        Returns:
            Dictionary with detailed similarity analysis
        """
        # Get section similarities
        section_similarities = self.calculate_section_similarities(
            doc1.get('sections', {}),
            doc2.get('sections', {})
        )
        
        # Define section weights based on document type
        # For resume-job matching, prioritize skills and experience
        section_weights = {
            'skills': 0.3,
            'experience': 0.25,
            'education': 0.15,
            'summary': 0.1,
            'projects': 0.1,
            'achievements': 0.1
        }
        
        # Calculate overall similarity
        overall_similarity = self.calculate_overall_similarity(
            doc1.get('sections', {}),
            doc2.get('sections', {}),
            section_weights
        )
        
        # Return comprehensive results
        return {
            'overall_similarity': overall_similarity,
            'section_similarities': section_similarities,
            'section_weights': section_weights
        }

"""
Semantic Analysis Module
Uses embeddings for contextual understanding.
"""

from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Tuple, Optional
import numpy as np
import spacy
from spacy.tokens import Doc
import re

class SemanticAnalyzer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        """
        Initialize the SemanticAnalyzer with the specified BERT model
        
        Args:
            model_name: Name of the BERT model to use (default: bert-base-uncased)
        """
        # Load BERT model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Load spaCy for NER
        self.nlp = spacy.load("en_core_web_sm")
        
        # Entity type mappings
        self.entity_groups = {
            'PERSON': 'People',
            'ORG': 'Organizations',
            'GPE': 'Locations',
            'LOC': 'Locations',
            'PRODUCT': 'Products',
            'EVENT': 'Events',
            'WORK_OF_ART': 'Creative Works',
            'LAW': 'Laws',
            'LANGUAGE': 'Languages',
            'DATE': 'Dates',
            'TIME': 'Times',
            'PERCENT': 'Percentages',
            'MONEY': 'Money',
            'QUANTITY': 'Quantities',
            'ORDINAL': 'Ordinals',
            'CARDINAL': 'Cardinals'
        }
    
    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Generate BERT embeddings for the given text
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array containing the embedding vector
        """
        # Add special tokens and convert to tensor
        encoded_input = self.tokenizer(text, padding=True, truncation=True, 
                                       max_length=512, return_tensors='pt')
        
        # Prevent gradient calculation for inference
        with torch.no_grad():
            # Get model output
            model_output = self.model(**encoded_input)
            
            # Use the [CLS] token embedding from the last hidden state
            sentence_embedding = model_output.last_hidden_state[:, 0, :].numpy()
        
        return sentence_embedding[0]  # Return as 1D array
    
    def get_contextual_embeddings(self, text: str) -> Dict[str, np.ndarray]:
        """
        Generate contextual embeddings for each sentence in the text
        
        Args:
            text: Input text to process
            
        Returns:
            Dictionary mapping sentences to their embeddings
        """
        # Basic sentence splitting
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        
        embeddings = {}
        for sentence in sentences:
            embeddings[sentence] = self.get_bert_embedding(sentence)
            
        return embeddings
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using cosine similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        # Get embeddings
        embedding1 = self.get_bert_embedding(text1)
        embedding2 = self.get_bert_embedding(text2)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(embedding1, embedding2)
        
        return similarity
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Prevent division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def calculate_section_similarities(self, 
                                      doc1_sections: Dict[str, str], 
                                      doc2_sections: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate similarities between corresponding sections of two documents
        
        Args:
            doc1_sections: Dictionary mapping section names to text for first document
            doc2_sections: Dictionary mapping section names to text for second document
            
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
            
            if text1 and text2:  # Ensure both sections have content
                similarities[section] = self.calculate_similarity(text1, text2)
            else:
                similarities[section] = 0.0
                
        return similarities
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract named entities from text using spaCy
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary mapping entity groups to lists of entities
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Initialize results
        entities_by_group = {}
        
        # Extract entities
        for ent in doc.ents:
            # Get entity group
            group = self.entity_groups.get(ent.label_, 'Other')
            
            # Initialize group if not already present
            if group not in entities_by_group:
                entities_by_group[group] = []
            
            # Add entity details
            entities_by_group[group].append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        return entities_by_group
    
    def extract_key_phrases(self, text: str, top_n: int = 5) -> List[Dict]:
        """
        Extract key phrases from text based on noun chunks and NER
        
        Args:
            text: Text to analyze
            top_n: Number of top phrases to return
            
        Returns:
            List of key phrases with relevance scores
        """
        # Process text with spaCy
        doc = self.nlp(text)
        
        phrases = []
        
        # Get noun chunks
        for chunk in doc.noun_chunks:
            # Skip chunks that are just pronouns or determiners
            if chunk.root.pos_ in ['PRON', 'DET']:
                continue
                
            # Skip very short chunks
            if len(chunk.text.split()) < 2:
                continue
            
            # Add noun chunk
            phrases.append({
                'text': chunk.text,
                'type': 'noun_chunk',
                'relevance': 0.7
            })
        
        # Add named entities with higher relevance
        for ent in doc.ents:
            phrases.append({
                'text': ent.text,
                'type': 'entity',
                'entity_type': ent.label_,
                'relevance': 0.9
            })
        
        # Remove duplicates (keep the one with higher relevance)
        unique_phrases = {}
        for phrase in phrases:
            key = phrase['text'].lower()
            if key not in unique_phrases or phrase['relevance'] > unique_phrases[key]['relevance']:
                unique_phrases[key] = phrase
        
        # Get top N phrases by relevance
        top_phrases = sorted(unique_phrases.values(), key=lambda x: x['relevance'], reverse=True)[:top_n]
        
        return top_phrases
    
    def analyze_semantic_context(self, 
                                text: str, 
                                context: Optional[str] = None) -> Dict:
        """
        Perform comprehensive semantic analysis on text
        
        Args:
            text: Main text to analyze
            context: Optional context text for comparison
            
        Returns:
            Dictionary with analysis results
        """
        results = {
            'entities': self.extract_entities(text),
            'key_phrases': self.extract_key_phrases(text)
        }
        
        # Calculate embedding for main text
        results['embedding'] = {
            'text': 'Embedding vector omitted for brevity'  # Don't include the actual vector in display results
        }
        
        # If context is provided, calculate similarity
        if context:
            results['context_similarity'] = self.calculate_similarity(text, context)
            results['context_entities'] = self.extract_entities(context)
            
            # Find shared entities
            shared_entities = []
            text_entities = [ent['text'].lower() for group in results['entities'].values() for ent in group]
            context_entities = [ent['text'].lower() for group in results['context_entities'].values() for ent in group]
            
            for entity in text_entities:
                if entity in context_entities:
                    shared_entities.append(entity)
            
            results['shared_entities'] = shared_entities
        
        return results

    def get_embeddings(self, text: str) -> np.ndarray:
        """Get text embeddings"""
        pass

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        pass

    def extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text"""
        pass

    def analyze_context(self, target: str, context: str) -> Dict[str, float]:
        """Analyze contextual relevance"""
        pass

    def find_similar_phrases(self, phrase: str, candidates: List[str]) -> List[Tuple[str, float]]:
        """Find semantically similar phrases"""
        pass

    def cluster_related_terms(self, terms: List[str]) -> Dict[str, List[str]]:
        """Cluster semantically related terms"""
        pass

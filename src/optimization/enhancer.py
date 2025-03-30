"""
Content Enhancement Module
Improves resume content quality and impact.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import json
import re
import requests
import numpy as np
from collections import Counter

# Import from other modules
from src.nlp.keyword_extractor import KeywordExtractor
from src.nlp.semantic_analyzer import SemanticAnalyzer
from src.preprocessing.cleaner import TextCleaner
from src.utils.constants import API_CONFIG

class ContentEnhancer:
    """
    Enhance resume content for better matching and readability
    - Bullet point enhancement
    - Keyword integration
    - Content rebalancing
    """
    
    def __init__(self,
                keyword_extractor: Optional[KeywordExtractor] = None,
                semantic_analyzer: Optional[SemanticAnalyzer] = None,
                text_cleaner: Optional[TextCleaner] = None,
                use_llm: bool = True,
                api_key: Optional[str] = None):
        """
        Initialize the content enhancer
        
        Args:
            keyword_extractor: Optional KeywordExtractor object
            semantic_analyzer: Optional SemanticAnalyzer object
            text_cleaner: Optional TextCleaner object
            use_llm: Whether to use LLM for enhanced content generation
            api_key: API key for LLM service (if use_llm is True)
        """
        self.keyword_extractor = keyword_extractor or KeywordExtractor(industry="software")
        self.semantic_analyzer = semantic_analyzer or SemanticAnalyzer()
        self.text_cleaner = text_cleaner or TextCleaner()
        self.use_llm = use_llm
        self.api_key = api_key or API_CONFIG.get("api_key", "")
        self.api_url = API_CONFIG.get("api_url", "")
        
        # Load action verbs for bullet points
        self.action_verbs = self._load_action_verbs()
        
        # Load section ideal word counts
        self.section_word_counts = {
            "summary": (50, 200),  # (min, max)
            "experience": (200, 500),
            "education": (50, 150),
            "skills": (100, 250),
            "projects": (100, 300),
            "achievements": (50, 150),
            "certifications": (30, 100)
        }
    
    def _load_action_verbs(self) -> Dict[str, List[str]]:
        """Load categorized action verbs for bullet points"""
        # In a real implementation, this would load from a file or database
        return {
            "leadership": [
                "led", "managed", "directed", "supervised", "coordinated", "headed",
                "spearheaded", "orchestrated", "oversaw", "guided", "mentored"
            ],
            "achievement": [
                "achieved", "exceeded", "improved", "increased", "reduced", "saved",
                "generated", "delivered", "boosted", "accelerated", "maximized"
            ],
            "analysis": [
                "analyzed", "assessed", "evaluated", "researched", "investigated",
                "examined", "reviewed", "surveyed", "studied", "identified"
            ],
            "development": [
                "developed", "created", "designed", "built", "implemented", "established",
                "launched", "initiated", "formulated", "constructed", "engineered"
            ],
            "communication": [
                "communicated", "presented", "authored", "wrote", "documented",
                "reported", "edited", "published", "facilitated", "negotiated"
            ],
            "technical": [
                "programmed", "coded", "engineered", "debugged", "deployed", 
                "integrated", "configured", "optimized", "maintained", "automated"
            ]
        }
    
    def enhance_bullet_points(self, 
                            bullet_points: List[str],
                            context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Enhance bullet points with better structure and impact
        
        Args:
            bullet_points: List of bullet point strings
            context: Optional context about job and skills
            
        Returns:
            List of enhanced bullet points with before/after comparison
        """
        # Initialize results
        enhanced_bullets = []
        
        # Extract job keywords if context is provided
        target_keywords = []
        if context and 'job_description' in context:
            job_keywords_tuples = self.keyword_extractor.extract_keywords(
                context['job_description'], 
                top_n=15
            )
            target_keywords = [kw.lower() for kw, _ in job_keywords_tuples]
        
        # Process each bullet point
        for bullet in bullet_points:
            # Clean the bullet point
            clean_bullet = self._clean_bullet_point(bullet)
            
            # Check if bullet starts with action verb
            starts_with_action = self._starts_with_action_verb(clean_bullet)
            
            # Check if bullet contains metrics
            contains_metrics = self._contains_metrics(clean_bullet)
            
            # Determine if bullet needs enhancement
            needs_enhancement = not starts_with_action or not contains_metrics
            
            # Check keyword presence
            bullet_keywords = set(re.findall(r'\b\w+\b', clean_bullet.lower()))
            matched_keywords = [kw for kw in target_keywords if kw in bullet_keywords]
            
            # Check for passive voice
            has_passive_voice = self._check_passive_voice(clean_bullet)
            
            # Enhance with traditional NLP if needed
            enhanced_bullet = clean_bullet
            method = "unchanged"
            
            if needs_enhancement:
                # Try rule-based enhancement first
                enhanced_bullet = self._rule_based_bullet_enhancement(
                    clean_bullet, 
                    starts_with_action,
                    contains_metrics,
                    has_passive_voice,
                    matched_keywords
                )
                method = "rule-based"
                
                # Use LLM for more complex cases
                if self.use_llm and (
                    has_passive_voice or 
                    not contains_metrics or 
                    len(matched_keywords) < 2
                ):
                    llm_enhanced = self._enhance_bullet_with_llm(
                        clean_bullet,
                        context,
                        starts_with_action,
                        contains_metrics
                    )
                    if llm_enhanced:
                        enhanced_bullet = llm_enhanced
                        method = "llm"
            
            # Add to results
            enhanced_bullets.append({
                'original': bullet,
                'enhanced': enhanced_bullet,
                'enhancement_method': method,
                'metrics_added': not contains_metrics and self._contains_metrics(enhanced_bullet),
                'action_verb_added': not starts_with_action and self._starts_with_action_verb(enhanced_bullet),
                'matched_keywords': matched_keywords
            })
        
        return enhanced_bullets
    
    def _clean_bullet_point(self, bullet: str) -> str:
        """Clean and normalize a bullet point"""
        # Remove bullet character if present
        cleaned = re.sub(r'^[\s•\-\*]+', '', bullet)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Ensure first letter is capitalized
        if cleaned and not cleaned[0].isupper():
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned
    
    def _starts_with_action_verb(self, text: str) -> bool:
        """Check if text starts with an action verb"""
        # Get first word
        match = re.match(r'^\s*(\w+)', text)
        if not match:
            return False
            
        first_word = match.group(1).lower()
        
        # Check if first word is in action verbs
        all_verbs = [verb for verbs in self.action_verbs.values() for verb in verbs]
        return first_word in all_verbs
    
    def _contains_metrics(self, text: str) -> bool:
        """Check if text contains metrics (numbers, percentages, etc.)"""
        # Check for numbers
        has_numbers = bool(re.search(r'\d+', text))
        
        # Check for percentages
        has_percentages = bool(re.search(r'\d+%', text))
        
        # Check for money values
        has_money = bool(re.search(r'[$€£]\d+|\d+\s(?:dollars|USD|EUR)', text))
        
        return has_numbers or has_percentages or has_money
    
    def _check_passive_voice(self, text: str) -> bool:
        """Check if text contains passive voice"""
        # Simple passive voice patterns
        passive_patterns = [
            r'\b(?:am|is|are|was|were|be|being|been)\s+\w+ed\b',
            r'\b(?:am|is|are|was|were|be|being|been)\s+\w+en\b'
        ]
        
        for pattern in passive_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
                
        return False
    
    def _rule_based_bullet_enhancement(self, 
                                     bullet: str, 
                                     has_action: bool,
                                     has_metrics: bool,
                                     has_passive: bool,
                                     target_keywords: List[str]) -> str:
        """Enhance bullet point using rule-based approach"""
        enhanced = bullet
        
        # Add action verb if missing
        if not has_action:
            # Determine appropriate verb category
            verb_category = 'achievement'  # Default
            
            if any(tech_word in bullet.lower() for tech_word in ['code', 'program', 'develop', 'algorithm', 'software']):
                verb_category = 'technical'
            elif any(analysis_word in bullet.lower() for analysis_word in ['research', 'study', 'analyze', 'review']):
                verb_category = 'analysis'
            elif any(lead_word in bullet.lower() for lead_word in ['team', 'group', 'direct', 'manage']):
                verb_category = 'leadership'
                
            # Get a verb from the category
            verbs = self.action_verbs.get(verb_category, self.action_verbs['achievement'])
            verb = verbs[0].capitalize()  # Just use the first one
            
            # Add the verb
            enhanced = f"{verb} {enhanced[0].lower() + enhanced[1:]}"
        
        # Try to convert passive to active
        if has_passive:
            # This is complex, in a real implementation would use NLP parsing
            # Here we'll just flag it for LLM enhancement
            pass
        
        # For metrics, we can't easily add them without real context
        # This would be handled by the LLM
        
        return enhanced
    
    def _enhance_bullet_with_llm(self, 
                               bullet: str,
                               context: Optional[Dict[str, Any]],
                               has_action: bool,
                               has_metrics: bool) -> Optional[str]:
        """Enhance bullet point using LLM"""
        if not self.api_key:
            return None
            
        try:
            # Prepare context string
            context_str = ""
            if context:
                if 'job_title' in context:
                    context_str += f"Job Title: {context['job_title']}\n"
                if 'job_keywords' in context:
                    context_str += f"Key Job Skills: {', '.join(context['job_keywords'][:5])}\n"
                if 'experience_level' in context:
                    context_str += f"Experience Level: {context['experience_level']}\n"
            
            # Create enhancement instructions
            instructions = []
            if not has_action:
                instructions.append("Start with a strong action verb")
            if not has_metrics:
                instructions.append("Add specific metrics or quantifiable results")
            if self._check_passive_voice(bullet):
                instructions.append("Convert from passive to active voice")
            
            instructions.append("Keep it concise (under 20 words)")
            
            # Create prompt for LLM
            prompt = f"""
            Enhance this resume bullet point:
            
            "{bullet}"
            
            {context_str}
            
            Enhancement needs:
            - {"\n- ".join(instructions)}
            
            Return ONLY the enhanced bullet point text with no additional explanation.
            """
            
            # Call LLM API
            enhanced_bullet = self._call_llm_api_text(prompt)
            
            # Clean the response
            if enhanced_bullet:
                enhanced_bullet = self._clean_bullet_point(enhanced_bullet)
                return enhanced_bullet
            
        except Exception as e:
            print(f"Error enhancing bullet point with LLM: {str(e)}")
        
        return None
    
    def integrate_keywords(self, 
                         section_text: str, 
                         target_keywords: List[Dict[str, Any]],
                         max_density: float = 0.04) -> Dict[str, Any]:
        """
        Integrate target keywords into text
        
        Args:
            section_text: Text to enhance with keywords
            target_keywords: List of target keywords with scores
            max_density: Maximum keyword density (as fraction)
            
        Returns:
            Dictionary with enhanced text and analysis
        """
        # Get text without current keywords
        original_words = self.text_cleaner.tokenize_text(section_text)
        word_count = len(original_words)
        
        # Extract existing keywords
        existing_keywords_tuples = self.keyword_extractor.extract_keywords(section_text, top_n=20)
        existing_keywords = [{'text': kw, 'score': score} for kw, score in existing_keywords_tuples]
        existing_kw_texts = [kw.lower() for kw, _ in existing_keywords_tuples]
        
        # Identify missing keywords
        missing_keywords = [
            kw for kw in target_keywords 
            if kw['text'].lower() not in existing_kw_texts
        ]
        
        # Sort by importance
        missing_keywords.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Calculate current keyword density
        current_matches = 0
        for kw in target_keywords:
            kw_text = kw['text'].lower()
            current_matches += len(re.findall(r'\b' + re.escape(kw_text) + r'\b', section_text.lower()))
        
        current_density = current_matches / word_count if word_count > 0 else 0
        
        # Determine how many keywords can be added
        available_slots = int(max_density * word_count) - current_matches
        available_slots = max(0, min(available_slots, len(missing_keywords)))
        
        # Select keywords to integrate
        keywords_to_add = missing_keywords[:available_slots]
        
        if not keywords_to_add:
            # No keywords needed or possible to add
            return {
                'enhanced_text': section_text,
                'current_density': current_density,
                'keywords_added': [],
                'enhancement_method': 'none'
            }
        
        # Try rule-based keyword integration
        rule_based_text = self._rule_based_keyword_integration(
            section_text, 
            keywords_to_add
        )
        
        # Use LLM for better integration if available
        if self.use_llm:
            llm_text = self._integrate_keywords_with_llm(
                section_text,
                keywords_to_add
            )
            
            if llm_text:
                # Verify keywords were actually added
                added_count = 0
                for kw in keywords_to_add:
                    kw_text = kw['text'].lower()
                    if re.search(r'\b' + re.escape(kw_text) + r'\b', llm_text.lower()) and \
                       not re.search(r'\b' + re.escape(kw_text) + r'\b', section_text.lower()):
                        added_count += 1
                
                # Use LLM text if it added keywords
                if added_count > 0:
                    return {
                        'enhanced_text': llm_text,
                        'current_density': current_density,
                        'target_density': max_density,
                        'keywords_added': [kw['text'] for kw in keywords_to_add[:added_count]],
                        'enhancement_method': 'llm'
                    }
        
        # Fall back to rule-based if LLM didn't work
        # Verify keywords were actually added
        added_count = 0
        for kw in keywords_to_add:
            kw_text = kw['text'].lower()
            if re.search(r'\b' + re.escape(kw_text) + r'\b', rule_based_text.lower()) and \
               not re.search(r'\b' + re.escape(kw_text) + r'\b', section_text.lower()):
                added_count += 1
        
        return {
            'enhanced_text': rule_based_text,
            'current_density': current_density,
            'target_density': max_density,
            'keywords_added': [kw['text'] for kw in keywords_to_add[:added_count]],
            'enhancement_method': 'rule-based'
        }
    
    def _rule_based_keyword_integration(self, 
                                      text: str, 
                                      keywords: List[Dict[str, Any]]) -> str:
        """Integrate keywords using rule-based approach"""
        if not keywords:
            return text
            
        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Add keywords to appropriate sentences
        keywords_added = 0
        
        for i, kw in enumerate(keywords):
            if keywords_added >= len(keywords):
                break
                
            kw_text = kw['text']
            
            # Skip if keyword already exists
            if re.search(r'\b' + re.escape(kw_text.lower()) + r'\b', text.lower()):
                continue
                
            # Find best sentence for this keyword
            best_sentence_idx = 0
            best_score = -1
            
            for j, sentence in enumerate(sentences):
                # Skip very short sentences
                if len(sentence.split()) < 5:
                    continue
                    
                # Calculate semantic similarity
                similarity = self.semantic_analyzer.calculate_similarity(sentence, kw_text)
                
                # Score based on similarity and position (prefer earlier sentences)
                position_factor = 1.0 - (j / len(sentences))
                score = similarity * 0.7 + position_factor * 0.3
                
                if score > best_score:
                    best_score = score
                    best_sentence_idx = j
            
            # Integrate keyword into the best sentence
            if best_score > 0:
                sentence = sentences[best_sentence_idx]
                
                # Simple integration - append to end
                if sentence.strip().endswith('.'):
                    # Add before the period
                    sentence = sentence[:-1] + f" including {kw_text}."
                else:
                    # Add with comma
                    sentence = sentence + f", including {kw_text}"
                    
                sentences[best_sentence_idx] = sentence
                keywords_added += 1
        
        # Rejoin text
        return ' '.join(sentences)
    
    def _integrate_keywords_with_llm(self, 
                                    text: str,
                                    keywords: List[Dict[str, Any]]) -> Optional[str]:
        """Integrate keywords into text using LLM"""
        if not self.api_key or not keywords:
            return None
        
        try:
            # Create keyword string
            keywords_str = ", ".join([kw['text'] for kw in keywords])
            
            # Create prompt for LLM
            prompt = f"""
            Integrate the following keywords naturally into the text below.
            The goal is to make the text more ATS-friendly for job applications.
            
            KEYWORDS: {keywords_str}
            
            TEXT:
            "{text}"
            
            Keep the overall meaning and style of the original text.
            DO NOT add extra explanations or formats - return ONLY the enhanced text.
            """
            
            # Call LLM API for text response
            enhanced_text = self._call_llm_api_text(prompt)
            
            if enhanced_text:
                # Return the enhanced text
                return enhanced_text
            
        except Exception as e:
            print(f"Error integrating keywords with LLM: {str(e)}")
        
        return None
    
    def rebalance_content(self, 
                        resume_sections: Dict[str, str],
                        target_job: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Rebalance content across resume sections
        
        Args:
            resume_sections: Dictionary mapping section names to content
            target_job: Optional target job for context
            
        Returns:
            Dictionary with rebalancing recommendations
        """
        # Analyze current section lengths
        section_analysis = {}
        total_words = 0
        
        for section, content in resume_sections.items():
            word_count = len(content.split())
            total_words += word_count
            
            # Get ideal range for this section
            ideal_range = self.section_word_counts.get(
                section.lower(), 
                (50, 200)  # Default range
            )
            
            # Determine status
            if word_count < ideal_range[0]:
                status = "too_short"
            elif word_count > ideal_range[1]:
                status = "too_long"
            else:
                status = "good"
                
            section_analysis[section] = {
                'word_count': word_count,
                'ideal_min': ideal_range[0],
                'ideal_max': ideal_range[1],
                'status': status
            }
        
        # Calculate current percentages
        for section in section_analysis:
            section_analysis[section]['percentage'] = (
                section_analysis[section]['word_count'] / total_words * 100
                if total_words > 0 else 0
            )
        
        # Determine priority sections based on job
        priority_sections = []
        if target_job:
            # Extract target job title and skills
            job_title = target_job.get('metadata', {}).get('title', '')
            
            # Simple rules for priority sections
            if any(tech in job_title.lower() for tech in ['developer', 'engineer', 'programmer']):
                priority_sections = ['Technical Skills', 'Experience', 'Projects']
            elif any(manager in job_title.lower() for manager in ['manager', 'director', 'lead']):
                priority_sections = ['Experience', 'Leadership', 'Achievements']
            elif any(analyst in job_title.lower() for analyst in ['analyst', 'consultant', 'specialist']):
                priority_sections = ['Skills', 'Experience', 'Education']
            else:
                # Default priorities
                priority_sections = ['Experience', 'Skills', 'Education']
        
        # Generate rebalancing recommendations
        recommendations = []
        
        # Identify sections to expand
        for section in priority_sections:
            if section in section_analysis and section_analysis[section]['status'] == "too_short":
                recommendations.append({
                    'section': section,
                    'action': 'expand',
                    'current_words': section_analysis[section]['word_count'],
                    'target_words': section_analysis[section]['ideal_min'],
                    'priority': 'high',
                    'reason': f"This is a priority section for the target job"
                })
        
        # Identify sections to trim
        for section, analysis in section_analysis.items():
            if analysis['status'] == "too_long":
                priority = 'medium'
                if section in priority_sections:
                    priority = 'low'  # Don't prioritize trimming important sections
                    
                recommendations.append({
                    'section': section,
                    'action': 'trim',
                    'current_words': analysis['word_count'],
                    'target_words': analysis['ideal_max'],
                    'priority': priority,
                    'reason': f"This section is longer than the recommended length"
                })
        
        # Identify low-priority sections to expand
        for section, analysis in section_analysis.items():
            if section not in priority_sections and analysis['status'] == "too_short":
                recommendations.append({
                    'section': section,
                    'action': 'expand',
                    'current_words': analysis['word_count'],
                    'target_words': analysis['ideal_min'],
                    'priority': 'low',
                    'reason': f"This section is shorter than the recommended length"
                })
        
        # Use LLM for specific content suggestions
        if self.use_llm:
            enhanced_recommendations = self._enhance_rebalancing_recommendations(
                recommendations,
                resume_sections,
                target_job
            )
            
            if enhanced_recommendations:
                recommendations = enhanced_recommendations
        
        # Sort recommendations by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order.get(x['priority'], 99))
        
        return {
            'section_analysis': section_analysis,
            'total_words': total_words,
            'recommendations': recommendations,
            'priority_sections': priority_sections
        }
    
    def _enhance_rebalancing_recommendations(self, 
                                          recommendations: List[Dict[str, Any]],
                                          resume_sections: Dict[str, str],
                                          target_job: Optional[Dict[str, Any]] = None) -> Optional[List[Dict[str, Any]]]:
        """Enhance rebalancing recommendations with LLM"""
        if not self.api_key or not recommendations:
            return None
            
        try:
            # Extract job details
            job_context = ""
            if target_job:
                job_title = target_job.get('metadata', {}).get('title', '')
                job_description = target_job.get('sections', {}).get('description', '')[:300]
                job_context = f"""
                Job Title: {job_title}
                Job Description: {job_description}
                """
            
            # Create prompt for LLM
            prompt = f"""
            {job_context}
            
            Here are content rebalancing recommendations for a resume:
            {json.dumps(recommendations, indent=2)}
            
            For each recommendation, add specific content suggestions that explain:
            1. Exactly what type of content to add or remove
            2. Examples of what would strengthen each section (if expanding)
            3. What to prioritize keeping or removing (if trimming)
            
            Return ONLY a JSON array of enhanced recommendations with the original fields plus a 'content_suggestions' field for each item.
            """
            
            # Call LLM API
            enhanced_recommendations = self._call_llm_api(prompt)
            
            # Process LLM response if valid
            if enhanced_recommendations and isinstance(enhanced_recommendations, list):
                return enhanced_recommendations
            
        except Exception as e:
            print(f"Error enhancing rebalancing recommendations with LLM: {str(e)}")
        
        return None
    
    def _call_llm_api(self, prompt: str) -> Any:
        """
        Call LLM API with the given prompt to get a JSON response
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Parsed JSON response or None if error
        """
        if not self.api_key or not self.api_url:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-4o-mini",  # Use the specified model
                "messages": [
                    {"role": "system", "content": "You are a helpful resume optimization assistant. Respond only with JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            print(f"Making API call to {self.api_url} with model {data['model']}")
            response = requests.post(self.api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                try:
                    # Extract JSON from response
                    response_json = response.json()
                    content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    # Parse JSON content - try different strategies
                    try:
                        # Try parsing the entire content as JSON first
                        return json.loads(content)
                    except json.JSONDecodeError:
                        # If that fails, try to find a JSON array in the content
                        match = re.search(r'\[(.*?)\]', content, re.DOTALL)
                        if match:
                            json_str = match.group(0)
                            return json.loads(json_str)
                        
                        # If still fails, try to find a JSON object
                        match = re.search(r'\{(.*?)\}', content, re.DOTALL)
                        if match:
                            json_str = "{" + match.group(1) + "}"
                            return json.loads(json_str)
                        
                        # If all parsing attempts fail, log and return None
                        print(f"Could not parse JSON from response: {content[:1000]}")
                        return None
                except Exception as e:
                    print(f"Error parsing API response: {str(e)}")
                    print(f"Raw response: {response.text[:1000]}")
                    return None
            else:
                print(f"API returned non-200 status code: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        
        except Exception as e:
            print(f"Error calling LLM API: {str(e)}")
            return None
    
    def _call_llm_api_text(self, prompt: str) -> Optional[str]:
        """
        Call LLM API with the given prompt to get a text response
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            Text response or None if error
        """
        if not self.api_key or not self.api_url:
            return None
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": "gpt-4o-mini",  # Use the specified model
                "messages": [
                    {"role": "system", "content": "You are a helpful resume optimization assistant. Provide only the requested content with no explanations or additional formatting."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7
            }
            
            print(f"Making API call to {self.api_url} with model {data['model']}")
            response = requests.post(self.api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                # Extract text from response
                try:
                    response_json = response.json()
                    content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    # Clean up the response
                    content = content.strip('"\'`').strip()
                    
                    return content
                except Exception as e:
                    print(f"Error parsing API response: {str(e)}")
                    print(f"Raw response: {response.text[:1000]}")  # Print first 1000 chars of response for debugging
                    return None
            else:
                print(f"API returned non-200 status code: {response.status_code}")
                print(f"Response: {response.text}")
                return None
        
        except Exception as e:
            print(f"Error calling LLM API for text: {str(e)}")
            return None

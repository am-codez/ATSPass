// Comprehensive synonym dictionary for common resume terms
const synonymDictionary = {
  // Technical Skills
  'programming': ['coding', 'development', 'software engineering', 'implementation'],
  'coding': ['programming', 'development', 'software engineering', 'implementation'],
  'development': ['programming', 'coding', 'software engineering', 'implementation'],
  'software': ['program', 'application', 'system', 'platform'],
  'program': ['software', 'application', 'system', 'platform'],
  'application': ['software', 'program', 'system', 'platform'],
  'system': ['software', 'program', 'application', 'platform'],
  'platform': ['software', 'program', 'application', 'system'],
  
  // Management & Leadership
  'manage': ['lead', 'direct', 'oversee', 'supervise', 'coordinate', 'administer'],
  'lead': ['manage', 'direct', 'oversee', 'supervise', 'coordinate', 'administer'],
  'direct': ['manage', 'lead', 'oversee', 'supervise', 'coordinate', 'administer'],
  'oversee': ['manage', 'lead', 'direct', 'supervise', 'coordinate', 'administer'],
  'supervise': ['manage', 'lead', 'direct', 'oversee', 'coordinate', 'administer'],
  'coordinate': ['manage', 'lead', 'direct', 'oversee', 'supervise', 'administer'],
  'administer': ['manage', 'lead', 'direct', 'oversee', 'supervise', 'coordinate'],
  
  // Analysis & Problem Solving
  'analyze': ['examine', 'evaluate', 'assess', 'review', 'investigate', 'scrutinize'],
  'examine': ['analyze', 'evaluate', 'assess', 'review', 'investigate', 'scrutinize'],
  'evaluate': ['analyze', 'examine', 'assess', 'review', 'investigate', 'scrutinize'],
  'assess': ['analyze', 'examine', 'evaluate', 'review', 'investigate', 'scrutinize'],
  'review': ['analyze', 'examine', 'evaluate', 'assess', 'investigate', 'scrutinize'],
  'investigate': ['analyze', 'examine', 'evaluate', 'assess', 'review', 'scrutinize'],
  'scrutinize': ['analyze', 'examine', 'evaluate', 'assess', 'review', 'investigate'],
  
  // Development & Creation
  'develop': ['create', 'build', 'design', 'implement', 'engineer', 'construct'],
  'create': ['develop', 'build', 'design', 'implement', 'engineer', 'construct'],
  'build': ['develop', 'create', 'design', 'implement', 'engineer', 'construct'],
  'design': ['develop', 'create', 'build', 'implement', 'engineer', 'construct'],
  'implement': ['develop', 'create', 'build', 'design', 'engineer', 'construct'],
  'engineer': ['develop', 'create', 'build', 'design', 'implement', 'construct'],
  'construct': ['develop', 'create', 'build', 'design', 'implement', 'engineer'],
  
  // Improvement & Enhancement
  'improve': ['enhance', 'optimize', 'refine', 'upgrade', 'advance', 'strengthen'],
  'enhance': ['improve', 'optimize', 'refine', 'upgrade', 'advance', 'strengthen'],
  'optimize': ['improve', 'enhance', 'refine', 'upgrade', 'advance', 'strengthen'],
  'refine': ['improve', 'enhance', 'optimize', 'upgrade', 'advance', 'strengthen'],
  'upgrade': ['improve', 'enhance', 'optimize', 'refine', 'advance', 'strengthen'],
  'advance': ['improve', 'enhance', 'optimize', 'refine', 'upgrade', 'strengthen'],
  'strengthen': ['improve', 'enhance', 'optimize', 'refine', 'upgrade', 'advance'],
  
  // Communication & Collaboration
  'communicate': ['convey', 'present', 'express', 'articulate', 'relay', 'transmit'],
  'convey': ['communicate', 'present', 'express', 'articulate', 'relay', 'transmit'],
  'present': ['communicate', 'convey', 'express', 'articulate', 'relay', 'transmit'],
  'express': ['communicate', 'convey', 'present', 'articulate', 'relay', 'transmit'],
  'articulate': ['communicate', 'convey', 'present', 'express', 'relay', 'transmit'],
  'relay': ['communicate', 'convey', 'present', 'express', 'articulate', 'transmit'],
  'transmit': ['communicate', 'convey', 'present', 'express', 'articulate', 'relay'],
  
  // Project Management
  'plan': ['organize', 'coordinate', 'arrange', 'schedule', 'strategize', 'prepare'],
  'organize': ['plan', 'coordinate', 'arrange', 'schedule', 'strategize', 'prepare'],
  'coordinate': ['plan', 'organize', 'arrange', 'schedule', 'strategize', 'prepare'],
  'arrange': ['plan', 'organize', 'coordinate', 'schedule', 'strategize', 'prepare'],
  'schedule': ['plan', 'organize', 'coordinate', 'arrange', 'strategize', 'prepare'],
  'strategize': ['plan', 'organize', 'coordinate', 'arrange', 'schedule', 'prepare'],
  'prepare': ['plan', 'organize', 'coordinate', 'arrange', 'schedule', 'strategize']
};

// Cache for synonyms to improve performance
const synonymCache = new Map();

// Get synonyms for a word
const getSynonyms = (word) => {
  // Check cache first
  if (synonymCache.has(word)) {
    return synonymCache.get(word);
  }

  const wordLower = word.toLowerCase();
  const synonyms = new Set();

  // Check if the word exists in our dictionary
  if (synonymDictionary[wordLower]) {
    synonymDictionary[wordLower].forEach(synonym => {
      synonyms.add(synonym);
    });
  }

  // Check if the word is a synonym of any other word
  Object.entries(synonymDictionary).forEach(([key, value]) => {
    if (value.includes(wordLower)) {
      synonyms.add(key);
    }
  });

  // Convert Set to Array and cache the result
  const synonymArray = Array.from(synonyms);
  synonymCache.set(word, synonymArray);
  return synonymArray;
};

// Get the best synonym based on context
const getBestSynonym = (word, targetKeywords) => {
  const synonyms = getSynonyms(word);
  
  // If no synonyms found, return the original word
  if (synonyms.length === 0) {
    return word;
  }

  // Score each synonym based on how well it matches target keywords
  const scoredSynonyms = synonyms.map(synonym => {
    let score = 0;
    
    // Check if the synonym is in target keywords
    if (targetKeywords.includes(synonym)) {
      score += 2;
    }
    
    // Check if any target keyword has this synonym
    for (const targetKeyword of targetKeywords) {
      const targetSynonyms = getSynonyms(targetKeyword);
      if (targetSynonyms.includes(synonym)) {
        score += 1;
      }
    }
    
    return { synonym, score };
  });

  // Sort by score and return the best match
  scoredSynonyms.sort((a, b) => b.score - a.score);
  return scoredSynonyms[0].synonym;
};

// Check if a word is a synonym of any target keyword
const isSynonymOfTarget = (word, targetKeywords) => {
  // Check if the word is directly in target keywords
  if (targetKeywords.includes(word)) {
    return true;
  }

  // Check if the word is a synonym of any target keyword
  for (const targetKeyword of targetKeywords) {
    const targetSynonyms = getSynonyms(targetKeyword);
    if (targetSynonyms.includes(word)) {
      return true;
    }
  }

  return false;
};

export {
  getSynonyms,
  getBestSynonym,
  isSynonymOfTarget
}; 
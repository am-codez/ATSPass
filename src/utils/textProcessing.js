import { getBestSynonym, isSynonymOfTarget } from './synonymUtils';

const stopwords = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
  'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that', 'the', 'this', 'to', 'was',
  'were', 'will', 'with', 'the', 'their', 'they', 'them', 'these', 'those',
  'am', 'been', 'being', 'have', 'had', 'having', 'do', 'does', 'did', 'doing',
  'would', 'should', 'could', 'shall', 'should', 'may', 'might', 'must'
]);

const synonymDictionary = {
  'develop': ['create', 'build', 'design', 'implement', 'engineer', 'construct'],
  'create': ['develop', 'build', 'design', 'implement', 'engineer', 'construct'],
  'build': ['develop', 'create', 'design', 'implement', 'engineer', 'construct'],
  'design': ['develop', 'create', 'build', 'implement', 'engineer', 'construct'],
  'implement': ['develop', 'create', 'build', 'design', 'engineer', 'construct'],
  'manage': ['lead', 'direct', 'oversee', 'supervise', 'coordinate', 'administer'],
  'lead': ['manage', 'direct', 'oversee', 'supervise', 'coordinate', 'administer'],
  'direct': ['manage', 'lead', 'oversee', 'supervise', 'coordinate', 'administer'],
  'oversee': ['manage', 'lead', 'direct', 'supervise', 'coordinate', 'administer'],
  'supervise': ['manage', 'lead', 'direct', 'oversee', 'coordinate', 'administer'],
  'analyze': ['examine', 'evaluate', 'assess', 'review', 'investigate', 'scrutinize'],
  'examine': ['analyze', 'evaluate', 'assess', 'review', 'investigate', 'scrutinize'],
  'evaluate': ['analyze', 'examine', 'assess', 'review', 'investigate', 'scrutinize'],
  'assess': ['analyze', 'examine', 'evaluate', 'review', 'investigate', 'scrutinize'],
  'review': ['analyze', 'examine', 'evaluate', 'assess', 'investigate', 'scrutinize'],
  'improve': ['enhance', 'optimize', 'refine', 'upgrade', 'advance', 'strengthen'],
  'enhance': ['improve', 'optimize', 'refine', 'upgrade', 'advance', 'strengthen'],
  'optimize': ['improve', 'enhance', 'refine', 'upgrade', 'advance', 'strengthen'],
  'refine': ['improve', 'enhance', 'optimize', 'upgrade', 'advance', 'strengthen'],
  'upgrade': ['improve', 'enhance', 'optimize', 'refine', 'advance', 'strengthen'],
  'advance': ['improve', 'enhance', 'optimize', 'refine', 'upgrade', 'strengthen'],
  'strengthen': ['improve', 'enhance', 'optimize', 'refine', 'upgrade', 'advance']
};

export const processText = (text) => {
  // Convert to lowercase and split into words
  const words = text.toLowerCase()
    .replace(/[^\w\s]/g, '') // Remove punctuation
    .split(/\s+/); // Split on whitespace

  // Filter out stopwords and short words
  const keywords = words.filter(word => 
    word.length > 2 && 
    !stopwords.has(word)
  );

  // Remove duplicates
  return [...new Set(keywords)];
};

export const findSynonyms = (word) => {
  return synonymDictionary[word.toLowerCase()] || [];
};

export const optimizeText = (text, targetKeywords) => {
  const sentences = text.split(/[.!?]+/).filter(s => s.trim());
  const optimizedSentences = [];
  let totalWordCount = 0;
  const originalWordCount = text.split(/\s+/).length;
  const maxWordIncrease = 50;

  for (const sentence of sentences) {
    const words = sentence.trim().split(/\s+/);
    const optimizedWords = [...words];
    let modified = false;
    let keywordCount = 0;

    // First pass: count keywords in the sentence
    for (const word of words) {
      if (isSynonymOfTarget(word.toLowerCase(), targetKeywords)) {
        keywordCount++;
      }
    }

    // Second pass: optimize the sentence
    for (let i = 0; i < words.length; i++) {
      const word = words[i].toLowerCase();
      
      // Check if the word is in target keywords or is a synonym
      if (isSynonymOfTarget(word, targetKeywords)) {
        const bestSynonym = getBestSynonym(word, targetKeywords);
        if (bestSynonym !== word) {
          optimizedWords[i] = bestSynonym;
          modified = true;
        }
      }
    }

    // Add relevant keywords if the sentence has few or none
    if (keywordCount < 2 && modified) {
      const relevantKeywords = targetKeywords.filter(keyword => 
        !optimizedWords.some(word => 
          isSynonymOfTarget(word.toLowerCase(), [keyword])
        )
      );

      if (relevantKeywords.length > 0) {
        const keywordToAdd = relevantKeywords[0];
        optimizedWords.push(keywordToAdd);
        totalWordCount++;
      }
    }

    // Only include modified sentences if we haven't exceeded the word limit
    const newWordCount = optimizedWords.join(' ').split(/\s+/).length;
    if (modified && (totalWordCount + newWordCount - words.length) <= originalWordCount + maxWordIncrease) {
      optimizedSentences.push(optimizedWords.join(' '));
      totalWordCount += newWordCount - words.length;
    } else {
      optimizedSentences.push(sentence.trim());
    }
  }

  return optimizedSentences.join('. ');
}; 
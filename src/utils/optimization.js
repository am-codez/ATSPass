import natural from 'natural';
import { findSynonyms } from './textProcessing';

const tokenizer = new natural.WordTokenizer();
const sentenceTokenizer = new natural.SentenceTokenizer();

export const optimizeResume = async (resumeText, jobKeywords) => {
  const sentences = sentenceTokenizer.tokenize(resumeText);
  const optimizedSentences = [];
  let totalWordCount = 0;
  const originalWordCount = tokenizer.tokenize(resumeText).length;
  const maxWordIncrease = 30;

  for (const sentence of sentences) {
    const words = tokenizer.tokenize(sentence);
    const optimizedWords = [...words];
    let modified = false;

    // Check each word in the sentence
    for (let i = 0; i < words.length; i++) {
      const word = words[i].toLowerCase();
      
      // If the word is in job keywords, try to find a better synonym
      if (jobKeywords.includes(word)) {
        const synonyms = findSynonyms(word);
        if (synonyms.length > 0) {
          // Choose a synonym that better matches the job description
          const bestSynonym = synonyms[0]; // In a real application, you might want to choose the best synonym based on context
          optimizedWords[i] = bestSynonym;
          modified = true;
        }
      }
    }

    // Only include modified sentences if we haven't exceeded the word limit
    const newWordCount = tokenizer.tokenize(optimizedWords.join(' ')).length;
    if (modified && (totalWordCount + newWordCount - words.length) <= originalWordCount + maxWordIncrease) {
      optimizedSentences.push(optimizedWords.join(' '));
      totalWordCount += newWordCount - words.length;
    } else {
      optimizedSentences.push(sentence);
    }
  }

  return optimizedSentences.join(' ');
};

export const validateOptimization = (originalText, optimizedText) => {
  const originalWords = tokenizer.tokenize(originalText).length;
  const optimizedWords = tokenizer.tokenize(optimizedText).length;
  
  if (optimizedWords > originalWords + 30) {
    throw new Error('Optimization exceeded maximum word count increase');
  }

  return true;
}; 
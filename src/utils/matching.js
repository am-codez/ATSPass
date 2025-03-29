import stringSimilarity from 'string-similarity';

export const calculateMatch = (jobKeywords, resumeKeywords) => {
  if (!jobKeywords.length || !resumeKeywords.length) {
    return 0;
  }

  let matchedKeywords = 0;
  const usedResumeKeywords = new Set();

  // First pass: exact matches
  for (const jobKeyword of jobKeywords) {
    for (const resumeKeyword of resumeKeywords) {
      if (jobKeyword === resumeKeyword && !usedResumeKeywords.has(resumeKeyword)) {
        matchedKeywords++;
        usedResumeKeywords.add(resumeKeyword);
        break;
      }
    }
  }

  // Second pass: similar matches (if not already matched)
  for (const jobKeyword of jobKeywords) {
    if (matchedKeywords === jobKeywords.length) break;

    for (const resumeKeyword of resumeKeywords) {
      if (!usedResumeKeywords.has(resumeKeyword)) {
        const similarity = stringSimilarity.compareTwoStrings(jobKeyword, resumeKeyword);
        if (similarity > 0.8) { // 80% similarity threshold
          matchedKeywords++;
          usedResumeKeywords.add(resumeKeyword);
          break;
        }
      }
    }
  }

  return (matchedKeywords / jobKeywords.length) * 100;
};

export const findMissingKeywords = (jobKeywords, resumeKeywords) => {
  const missingKeywords = [];
  const usedResumeKeywords = new Set();

  // First pass: exact matches
  for (const jobKeyword of jobKeywords) {
    let found = false;
    for (const resumeKeyword of resumeKeywords) {
      if (jobKeyword === resumeKeyword) {
        found = true;
        usedResumeKeywords.add(resumeKeyword);
        break;
      }
    }
    if (!found) {
      missingKeywords.push(jobKeyword);
    }
  }

  // Second pass: similar matches
  for (const jobKeyword of jobKeywords) {
    if (missingKeywords.includes(jobKeyword)) {
      for (const resumeKeyword of resumeKeywords) {
        if (!usedResumeKeywords.has(resumeKeyword)) {
          const similarity = stringSimilarity.compareTwoStrings(jobKeyword, resumeKeyword);
          if (similarity > 0.8) {
            missingKeywords.splice(missingKeywords.indexOf(jobKeyword), 1);
            usedResumeKeywords.add(resumeKeyword);
            break;
          }
        }
      }
    }
  }

  return missingKeywords;
}; 
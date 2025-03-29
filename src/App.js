import React, { useState } from 'react';
import './App.css';
import { processText, optimizeText } from './utils/textProcessing';
import { processResume } from './utils/resumeProcessing';
import { calculateMatch } from './utils/matching';

function App() {
  const [jobDescription, setJobDescription] = useState('');
  const [resumeFile, setResumeFile] = useState(null);
  const [initialMatch, setInitialMatch] = useState(null);
  const [finalMatch, setFinalMatch] = useState(null);
  const [optimizedResume, setOptimizedResume] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [debugInfo, setDebugInfo] = useState(null);

  const handleJobDescriptionChange = (e) => {
    setJobDescription(e.target.value);
    setError(null);
  };

  const handleResumeUpload = (e) => {
    setResumeFile(e.target.files[0]);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setDebugInfo(null);

    try {
      console.log('Starting analysis...');
      
      // Process job description
      console.log('Processing job description...');
      const jobKeywords = processText(jobDescription);
      console.log('Job keywords:', jobKeywords);
      
      // Process resume
      console.log('Processing resume...');
      const resumeText = await processResume(resumeFile);
      console.log('Resume text:', resumeText);
      
      const resumeKeywords = processText(resumeText);
      console.log('Resume keywords:', resumeKeywords);

      // Calculate initial match
      console.log('Calculating initial match...');
      const initialMatchPercentage = calculateMatch(jobKeywords, resumeKeywords);
      console.log('Initial match:', initialMatchPercentage);
      setInitialMatch(initialMatchPercentage);

      // Optimize resume
      console.log('Optimizing resume...');
      const optimizedText = optimizeText(resumeText, jobKeywords);
      console.log('Optimized text:', optimizedText);
      
      const optimizedKeywords = processText(optimizedText);
      console.log('Optimized keywords:', optimizedKeywords);
      
      const finalMatchPercentage = calculateMatch(jobKeywords, optimizedKeywords);
      console.log('Final match:', finalMatchPercentage);
      
      setFinalMatch(finalMatchPercentage);
      setOptimizedResume(optimizedText);

      // Set debug information
      setDebugInfo({
        jobKeywords,
        resumeKeywords,
        optimizedKeywords,
        originalText: resumeText,
        optimizedText
      });

    } catch (error) {
      console.error('Error processing files:', error);
      setError(error.message || 'An error occurred while processing the files.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Resume Optimizer</h1>
      </header>
      <main>
        <form onSubmit={handleSubmit}>
          <div className="input-section">
            <h2>Job Description</h2>
            <textarea
              value={jobDescription}
              onChange={handleJobDescriptionChange}
              placeholder="Paste the job description here..."
              rows="10"
              required
            />
          </div>

          <div className="input-section">
            <h2>Resume</h2>
            <input
              type="file"
              accept=".pdf"
              onChange={handleResumeUpload}
              required
            />
          </div>

          <button type="submit" disabled={loading}>
            {loading ? 'Processing...' : 'Analyze and Optimize'}
          </button>
        </form>

        {error && (
          <div className="error-section">
            <p className="error-message">{error}</p>
          </div>
        )}

        {debugInfo && (
          <div className="debug-section">
            <h2>Debug Information</h2>
            <div className="debug-content">
              <div className="debug-item">
                <h3>Job Keywords</h3>
                <pre>{JSON.stringify(debugInfo.jobKeywords, null, 2)}</pre>
              </div>
              <div className="debug-item">
                <h3>Original Resume Keywords</h3>
                <pre>{JSON.stringify(debugInfo.resumeKeywords, null, 2)}</pre>
              </div>
              <div className="debug-item">
                <h3>Optimized Resume Keywords</h3>
                <pre>{JSON.stringify(debugInfo.optimizedKeywords, null, 2)}</pre>
              </div>
              <div className="debug-item">
                <h3>Original Text</h3>
                <pre>{debugInfo.originalText}</pre>
              </div>
              <div className="debug-item">
                <h3>Optimized Text</h3>
                <pre>{debugInfo.optimizedText}</pre>
              </div>
            </div>
          </div>
        )}

        {initialMatch !== null && (
          <div className="results-section">
            <h2>Results</h2>
            <p>Initial Match: {initialMatch.toFixed(2)}%</p>
            {finalMatch !== null && (
              <p>Final Match: {finalMatch.toFixed(2)}%</p>
            )}
            {optimizedResume && (
              <div className="optimized-resume">
                <h3>Optimized Resume</h3>
                <div className="resume-content">
                  {optimizedResume.split('\n\n').map((section, sectionIndex) => (
                    <div key={sectionIndex} className="resume-section">
                      {section.split('\n').map((line, lineIndex) => {
                        const trimmedLine = line.trim();
                        // Check if this is a header (short line or contains common header indicators)
                        const isHeader = trimmedLine.length < 50 || 
                          /^(EDUCATION|EXPERIENCE|SKILLS|PROJECTS|CERTIFICATIONS|SUMMARY|OBJECTIVE|PROFILE|QUALIFICATIONS|WORK|EMPLOYMENT|PROFESSIONAL|ACADEMIC|TECHNICAL)/i.test(trimmedLine);
                        
                        return (
                          <p 
                            key={lineIndex} 
                            className={isHeader ? 'resume-header' : 'resume-line'}
                            style={{ 
                              textAlign: trimmedLine.length < 30 ? 'center' : 'left',
                              marginLeft: trimmedLine.length < 30 ? '0' : '20px'
                            }}
                          >
                            {trimmedLine}
                          </p>
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App; 
// PDF.js library for PDF processing
const pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

document.addEventListener('DOMContentLoaded', function () {
    // Get DOM elements
    const resumeText = document.getElementById('resume-text');
    const resumeFile = document.getElementById('resume-file');
    const jobDescFile = document.getElementById('job-desc-file');
    const jobDescription = document.getElementById('job-description');
    const analyzeBtn = document.getElementById('analyze-btn');
    const matchScore = document.getElementById('match-score');
    const missingKeywords = document.getElementById('missing-keywords');
    const enhancedScore = document.getElementById('enhanced-score');
    const addedKeywords = document.getElementById('added-keywords');
    const enhancedResumeText = document.getElementById('enhanced-resume-text');

    // Handle resume PDF upload
    if (resumeFile) {
        resumeFile.addEventListener('change', async function (event) {
            const file = event.target.files[0];
            if (!file) return;

            console.log("Processing resume PDF file:", file.name);

            try {
                const arrayBuffer = await file.arrayBuffer();
                const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
                let text = '';

                for (let i = 1; i <= pdf.numPages; i++) {
                    const page = await pdf.getPage(i);
                    const textContent = await page.getTextContent();
                    text += textContent.items.map(item => item.str).join(' ') + '\n';
                }

                resumeText.value = text;
                console.log("Resume PDF processed successfully");
            } catch (error) {
                console.error('Error processing resume PDF:', error);
                alert('Error processing resume PDF file. Please try again.');
            }
        });
    }

    // Handle job description PDF upload
    if (jobDescFile) {
        jobDescFile.addEventListener('change', async function (event) {
            const file = event.target.files[0];
            if (!file) return;

            console.log("Processing job description PDF file:", file.name);

            try {
                const arrayBuffer = await file.arrayBuffer();
                const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
                let text = '';

                for (let i = 1; i <= pdf.numPages; i++) {
                    const page = await pdf.getPage(i);
                    const textContent = await page.getTextContent();
                    text += textContent.items.map(item => item.str).join(' ') + '\n';
                }

                jobDescription.value = text;
                console.log("Job description PDF processed successfully");
            } catch (error) {
                console.error('Error processing job description PDF:', error);
                alert('Error processing job description PDF file. Please try again.');
            }
        });
    }

    // Handle analyze button click
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async function () {
            const resume = resumeText.value;
            const jobDesc = jobDescription.value;

            if (!resume || !jobDesc) {
                alert('Please provide both resume and job description');
                return;
            }

            console.log("Analyzing resume and job description");

            try {
                // Show loading state
                analyzeBtn.disabled = true;
                analyzeBtn.textContent = 'Analyzing...';

                // Create form data with the correct structure
                const formData = new FormData();
                formData.append('resume_text', resume);
                formData.append('job_description', jobDesc);

                // Use relative URL for API endpoint
                console.log("Sending data to backend...");
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const errorData = await response.text();
                    console.error("Server error:", errorData);
                    throw new Error(errorData || 'Analysis failed');
                }

                const result = await response.json();
                console.log("Analysis results received:", result);

                // Update match score
                if (matchScore && result.match_score !== undefined) {
                    matchScore.textContent = `${result.match_score}%`;
                }

                // Update missing keywords
                if (missingKeywords) {
                    missingKeywords.innerHTML = '';
                    if (result.missing_keywords && result.missing_keywords.length > 0) {
                        result.missing_keywords.forEach(keyword => {
                            const keywordSpan = document.createElement('span');
                            keywordSpan.className = 'keyword';
                            keywordSpan.textContent = keyword;
                            missingKeywords.appendChild(keywordSpan);
                        });
                    } else {
                        missingKeywords.textContent = 'None found';
                    }
                }

                // Update enhanced tab data if available
                if (enhancedScore && result.enhanced_score !== undefined) {
                    enhancedScore.textContent = `${result.enhanced_score}%`;
                }

                // Update added keywords
                if (addedKeywords) {
                    addedKeywords.innerHTML = '';
                    if (result.added_keywords && result.added_keywords.length > 0) {
                        result.added_keywords.forEach(keyword => {
                            const keywordSpan = document.createElement('span');
                            keywordSpan.className = 'keyword';
                            keywordSpan.textContent = keyword;
                            addedKeywords.appendChild(keywordSpan);
                        });
                    } else {
                        addedKeywords.textContent = 'None added';
                    }
                }

                // Update enhanced resume text
                if (enhancedResumeText && result.enhanced_resume) {
                    enhancedResumeText.textContent = result.enhanced_resume;
                }

                // Show the analysis tab
                showTab('analysis');
            } catch (error) {
                console.error('Error:', error);
                alert(error.message || 'Failed to analyze resume. Please try again.');
            } finally {
                // Reset button state
                analyzeBtn.disabled = false;
                analyzeBtn.textContent = 'Analyze Match';
            }
        });
    }

    // Initialize tab functionality
    const tabButtons = document.querySelectorAll('.tab-button');
    tabButtons.forEach(button => {
        button.addEventListener('click', function () {
            const tabId = button.getAttribute('data-tab');
            if (tabId) {
                showTab(tabId);
            }
        });
    });

    // Also handle tab switching on enhance button
    const enhanceButtons = document.querySelectorAll('.enhance-button');
    enhanceButtons.forEach(button => {
        button.addEventListener('click', function () {
            const tabId = button.getAttribute('data-tab');
            if (tabId) {
                showTab(tabId);
            }
        });
    });
});

function showTab(tabId) {
    console.log("Switching to tab:", tabId);

    // Update counter
    const tabIndex = ['input', 'analysis', 'enhanced'].indexOf(tabId) + 1;
    const tabCounter = document.querySelector('.tab-counter');
    if (tabCounter) {
        tabCounter.textContent = `${tabIndex}/3`;
    }

    // Update active tab button
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
        if (button.getAttribute('data-tab') === tabId) {
            button.classList.add('active');
        }
    });

    // Show active content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    const activeTab = document.getElementById(tabId);
    if (activeTab) {
        activeTab.classList.add('active');
    }
} 
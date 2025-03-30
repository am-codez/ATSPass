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

    // New UI elements
    const scoreMeterFill = document.getElementById('score-meter-fill');
    const enhancedScoreMeterFill = document.getElementById('enhanced-score-meter-fill');
    const recommendationsList = document.getElementById('recommendations-list');
    const skillsGapMeter = document.getElementById('skills-gap-meter');
    const experienceGapMeter = document.getElementById('experience-gap-meter');
    const educationGapMeter = document.getElementById('education-gap-meter');
    const optimizationSummary = document.getElementById('optimization-summary');
    const enhancedBullets = document.getElementById('enhanced-bullets');
    const atsTipsList = document.getElementById('ats-tips-list');
    const toggleDiffBtn = document.getElementById('toggle-diff-btn');
    const downloadResumeBtn = document.getElementById('download-resume-btn');

    // Error popup elements
    const closeButton = document.getElementById('close-button');
    const errorPopup = document.getElementById('error-popup');
    const errorOkBtn = document.getElementById('error-ok-btn');

    // Handle close button click - show error popup
    if (closeButton) {
        closeButton.addEventListener('click', function (event) {
            event.preventDefault();
            event.stopPropagation();
            errorPopup.classList.add('show');
        });
    }

    // Handle OK button click on error popup
    if (errorOkBtn) {
        errorOkBtn.addEventListener('click', function () {
            errorPopup.classList.remove('show');
        });
    }

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

                const result = await response.json();

                if (!response.ok) {
                    console.error("Server error:", result.error);
                    throw new Error(result.error || 'Analysis failed');
                }

                console.log("Analysis results received:", result);

                // Update UI with results
                updateAnalysisUI(result);

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

    // Toggle diff view for enhanced resume
    if (toggleDiffBtn) {
        toggleDiffBtn.addEventListener('click', function () {
            const isShowingDiff = toggleDiffBtn.textContent === 'Hide Changes';

            if (isShowingDiff) {
                // Switch to clean view
                toggleDiffBtn.textContent = 'Show Changes';
                renderCleanResume();
            } else {
                // Switch to diff view
                toggleDiffBtn.textContent = 'Hide Changes';
                renderDiffResume();
            }
        });
    }

    // Download enhanced resume
    if (downloadResumeBtn) {
        downloadResumeBtn.addEventListener('click', function () {
            const resumeContent = enhancedResumeText.textContent;
            if (!resumeContent) return;

            const blob = new Blob([resumeContent], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);

            const a = document.createElement('a');
            a.href = url;
            a.download = 'enhanced_resume.txt';
            document.body.appendChild(a);
            a.click();

            // Cleanup
            setTimeout(() => {
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            }, 100);
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
                // If going to enhance tab, fetch enhancements first
                if (tabId === 'enhanced' && !window.enhancementsLoaded) {
                    getEnhancements();
                } else {
                    showTab(tabId);
                }
            }
        });
    });

    // Function to update analysis UI with results
    function updateAnalysisUI(result) {
        // Update match score
        if (matchScore && result.match_score !== undefined) {
            const score = parseFloat(result.match_score);
            matchScore.textContent = `${score.toFixed(1)}%`;

            // Update score meter
            if (scoreMeterFill) {
                scoreMeterFill.style.width = `${score}%`;
            }
        }

        // Update missing keywords
        if (missingKeywords) {
            missingKeywords.innerHTML = '';
            if (result.missing_keywords && result.missing_keywords.length > 0) {
                result.missing_keywords.forEach(keyword => {
                    const keywordSpan = document.createElement('span');
                    keywordSpan.className = 'keyword';

                    // Handle both string and object formats
                    if (typeof keyword === 'string') {
                        keywordSpan.textContent = keyword;
                    } else if (typeof keyword === 'object') {
                        // Get the keyword text from either 'text', 'name', or 'skill' property
                        const keywordText = keyword.text || keyword.name || keyword.skill || JSON.stringify(keyword);
                        keywordSpan.textContent = keywordText;
                    }

                    missingKeywords.appendChild(keywordSpan);
                });
            } else {
                missingKeywords.textContent = 'None found';
            }
        }

        // Update gap analysis meters
        if (result.gap_analysis) {
            const { skills_match, experience_match, education_match } = result.gap_analysis;

            if (skillsGapMeter && skills_match !== undefined) {
                skillsGapMeter.style.width = `${skills_match}%`;
            }

            if (experienceGapMeter && experience_match !== undefined) {
                experienceGapMeter.style.width = `${experience_match}%`;
            }

            if (educationGapMeter && education_match !== undefined) {
                educationGapMeter.style.width = `${education_match}%`;
            }
        }

        // Update recommendations
        if (recommendationsList && result.recommendations) {
            recommendationsList.innerHTML = '';

            result.recommendations.forEach(rec => {
                const recItem = document.createElement('div');
                recItem.className = `recommendation-item ${rec.priority || 'medium'}-priority`;

                const recTitle = document.createElement('div');
                recTitle.className = 'recommendation-title';
                recTitle.textContent = rec.title;

                const recDesc = document.createElement('div');
                recDesc.className = 'recommendation-description';
                recDesc.textContent = rec.description;

                recItem.appendChild(recTitle);
                recItem.appendChild(recDesc);
                recommendationsList.appendChild(recItem);
            });

            if (result.recommendations.length === 0) {
                recommendationsList.textContent = 'No recommendations found';
            }
        }
    }

    // Function to get enhancements from the server
    async function getEnhancements() {
        try {
            const resume = resumeText.value;
            const jobDesc = jobDescription.value;

            if (!resume || !jobDesc) {
                alert('Please provide both resume and job description');
                return;
            }

            // Create form data
            const formData = new FormData();
            formData.append('resume_text', resume);
            formData.append('job_description', jobDesc);

            console.log("Fetching enhancements...");
            const response = await fetch('/api/enhance', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                console.error("Server error:", result.error);
                throw new Error(result.error || 'Enhancement failed');
            }

            console.log("Enhancement results received:", result);

            // Update UI with enhancement results
            updateEnhancementsUI(result);

            // Mark enhancements as loaded
            window.enhancementsLoaded = true;

            // Show the enhanced tab
            showTab('enhanced');
        } catch (error) {
            console.error('Error:', error);
            alert(error.message || 'Failed to enhance resume. Please try again.');
        }
    }

    // Function to update enhancements UI
    function updateEnhancementsUI(result) {
        // Update enhanced score
        if (enhancedScore && result.enhanced_score !== undefined) {
            const score = parseFloat(result.enhanced_score);
            enhancedScore.textContent = `${score.toFixed(1)}%`;

            // Update enhanced score meter
            if (enhancedScoreMeterFill) {
                enhancedScoreMeterFill.style.width = `${score}%`;
            }
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

        // Update optimization summary
        if (optimizationSummary && result.optimization_summary) {
            optimizationSummary.innerHTML = '';

            const summary = result.optimization_summary;
            const stats = [
                { label: 'Keyword Match', value: `${summary.keyword_match_increase || 0}%` },
                { label: 'Bullets Enhanced', value: `${summary.bullets_enhanced || 0}` },
                { label: 'Sections Optimized', value: `${summary.sections_improved || 0}` },
                { label: 'Score Improvement', value: `+${summary.score_improvement || 0}%` }
            ];

            stats.forEach(stat => {
                const statItem = document.createElement('div');
                statItem.className = 'stat-item';

                const label = document.createElement('span');
                label.className = 'stat-label';
                label.textContent = stat.label;

                const value = document.createElement('span');
                value.className = 'stat-value';
                value.textContent = stat.value;

                statItem.appendChild(label);
                statItem.appendChild(value);
                optimizationSummary.appendChild(statItem);
            });
        }

        // Update enhanced bullets
        if (enhancedBullets && result.enhanced_bullets) {
            enhancedBullets.innerHTML = '';

            if (result.enhanced_bullets.length > 0) {
                result.enhanced_bullets.forEach(bullet => {
                    const bulletDiv = document.createElement('div');
                    bulletDiv.className = 'bullet-comparison';

                    // Original bullet
                    const originalDiv = document.createElement('div');

                    const originalLabel = document.createElement('div');
                    originalLabel.className = 'bullet-label';
                    originalLabel.textContent = 'Original';

                    const originalText = document.createElement('div');
                    originalText.className = 'bullet-original';
                    originalText.textContent = bullet.original || '';

                    originalDiv.appendChild(originalLabel);
                    originalDiv.appendChild(originalText);

                    // Enhanced bullet
                    const enhancedDiv = document.createElement('div');

                    const enhancedLabel = document.createElement('div');
                    enhancedLabel.className = 'bullet-label';
                    enhancedLabel.textContent = 'Enhanced';

                    const enhancedText = document.createElement('div');
                    enhancedText.className = 'bullet-enhanced';
                    enhancedText.textContent = bullet.enhanced || '';

                    enhancedDiv.appendChild(enhancedLabel);
                    enhancedDiv.appendChild(enhancedText);

                    // Added keywords if any
                    if (bullet.matched_keywords && bullet.matched_keywords.length > 0) {
                        const keywordsDiv = document.createElement('div');
                        keywordsDiv.className = 'bullet-keywords';

                        bullet.matched_keywords.forEach(keyword => {
                            const keywordSpan = document.createElement('span');
                            keywordSpan.className = 'bullet-keyword';
                            keywordSpan.textContent = keyword;
                            keywordsDiv.appendChild(keywordSpan);
                        });

                        enhancedDiv.appendChild(keywordsDiv);
                    }

                    bulletDiv.appendChild(originalDiv);
                    bulletDiv.appendChild(enhancedDiv);
                    enhancedBullets.appendChild(bulletDiv);
                });
            } else {
                enhancedBullets.textContent = 'No bullet points enhanced';
            }
        }

        // Update ATS tips
        if (atsTipsList && result.ats_tips) {
            atsTipsList.innerHTML = '';

            if (result.ats_tips.length > 0) {
                result.ats_tips.forEach(tip => {
                    const tipDiv = document.createElement('div');
                    tipDiv.className = 'ats-tip';

                    const tipTitle = document.createElement('div');
                    tipTitle.className = 'ats-tip-title';
                    tipTitle.textContent = tip.title || '';

                    const tipDesc = document.createElement('div');
                    tipDesc.className = 'ats-tip-description';
                    tipDesc.textContent = tip.description || '';

                    tipDiv.appendChild(tipTitle);
                    tipDiv.appendChild(tipDesc);
                    atsTipsList.appendChild(tipDiv);
                });
            } else {
                atsTipsList.textContent = 'No ATS tips available';
            }
        }

        // Update enhanced resume text
        if (enhancedResumeText && result.enhanced_resume) {
            // Store the original and enhanced text for diff view
            window.originalResumeText = resumeText.value;
            window.enhancedResumeText = result.enhanced_resume;

            // Default to showing clean version
            enhancedResumeText.textContent = result.enhanced_resume;
        }
    }
});

// Function to switch between tabs
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

// Render functions for enhanced resume
function renderDiffResume() {
    const enhancedResumeText = document.getElementById('enhanced-resume-text');
    if (!window.originalResumeText || !window.enhancedResumeText || !enhancedResumeText) return;

    // In a real implementation, we would use a diff algorithm library
    // For simplicity, we'll just highlight the entire enhanced text
    enhancedResumeText.innerHTML = '';

    const diffSpan = document.createElement('span');
    diffSpan.className = 'diff-added';
    diffSpan.textContent = window.enhancedResumeText;

    enhancedResumeText.appendChild(diffSpan);
}

function renderCleanResume() {
    const enhancedResumeText = document.getElementById('enhanced-resume-text');
    if (!window.enhancedResumeText || !enhancedResumeText) return;

    enhancedResumeText.textContent = window.enhancedResumeText;
}

// Add event listener for the standalone ENHANCE RESUME button on the analysis page
document.addEventListener('DOMContentLoaded', function () {
    const standaloneEnhanceBtn = document.querySelector('.button-container button.enhance-button');
    if (standaloneEnhanceBtn) {
        standaloneEnhanceBtn.addEventListener('click', function () {
            // This is the large red ENHANCE RESUME button at the bottom of the analysis page
            console.log("Standalone enhance button clicked");
            getEnhancements();
        });
    }
});
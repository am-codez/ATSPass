// PDF.js library for PDF processing
const pdfjsLib = window['pdfjs-dist/build/pdf'];
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

// Groq API configuration
const GROQ_API_KEY = 'YOUR_GROQ_API_KEY'; // Replace with your actual Groq API key
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';

// DOM Elements
const resumeText = document.getElementById('resume-text');
const resumeFile = document.getElementById('resume-file');
const jobDescription = document.getElementById('job-description');
const analyzeBtn = document.getElementById('analyze-btn');
const processedResume = document.getElementById('processed-resume');
const matchScore = document.getElementById('match-score');
const incorporatedWords = document.getElementById('incorporated-words');

// Event Listeners
resumeFile.addEventListener('change', handlePDFUpload);
analyzeBtn.addEventListener('click', analyzeResume);

async function handlePDFUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

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
    } catch (error) {
        console.error('Error processing PDF:', error);
        alert('Error processing PDF file. Please try again.');
    }
}

async function analyzeResume() {
    const resume = resumeText.value;
    const jobDesc = jobDescription.value;

    if (!resume || !jobDesc) {
        alert('Please provide both resume and job description.');
        return;
    }

    try {
        // Show loading state
        processedResume.textContent = 'Processing...';
        analyzeBtn.disabled = true;

        // TODO: Add actual API call here
        // For now, just show a placeholder
        processedResume.textContent = 'Resume analysis would happen here...';
        matchScore.textContent = '85%';
        updateIncorporatedWords(['example', 'keywords', 'here']);
    } catch (error) {
        console.error('Error analyzing resume:', error);
        alert('Error analyzing resume. Please try again.');
    } finally {
        analyzeBtn.disabled = false;
    }
}

function updateIncorporatedWords(words) {
    incorporatedWords.innerHTML = words
        .map(word => `<span>${word}</span>`)
        .join('');
}

function showTab(tabId) {
    // Update counter
    const tabIndex = ['input', 'analysis', 'enhanced'].indexOf(tabId) + 1;
    document.querySelector('.tab-counter').textContent = `${tabIndex}/3`;
    
    // Update active tab button
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
        if(button.getAttribute('data-tab') === tabId) {
            button.classList.add('active');
        }
    });
    
    // Show active content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tabId).classList.add('active');
}

document.addEventListener('DOMContentLoaded', function() {
    // Get all tab buttons
    const tabButtons = document.querySelectorAll('.tab-button');
    
    // Add click listeners to each tab button
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            showTab(tabId);
        });
    });

    // Get analyze and enhance buttons
    const analyzeButton = document.querySelector('.analyze-button');
    const enhanceButton = document.querySelector('.enhance-button');

    // Add click listeners to action buttons
    analyzeButton.addEventListener('click', () => {
        showTab('analysis');
    });

    enhanceButton.addEventListener('click', () => {
        showTab('enhanced');
    });
}); 
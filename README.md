# Resume Optimizer

A React-based application that helps optimize your resume for specific job descriptions by analyzing keyword matches and suggesting improvements.

## Features

- Upload and analyze PDF resumes
- Process job descriptions to extract relevant keywords
- Calculate initial match percentage between resume and job description
- Optimize resume content using synonym replacement
- Maintain professional tone while improving keyword matches
- Generate optimized PDF output

## Prerequisites

- Node.js (v14 or higher)
- npm (v6 or higher)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd hackapp
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will open in your default browser at `http://localhost:3000`.

## Usage

1. Paste the job description into the text area
2. Upload your resume as a PDF file
3. Click "Analyze and Optimize"
4. View the initial match percentage
5. Wait for the optimization process to complete
6. View the final match percentage
7. Download the optimized resume

## Technical Details

The application uses the following libraries:
- `natural` for text processing and tokenization
- `pdf-lib` for PDF handling
- `string-similarity` for text comparison
- React for the user interface

## Limitations

- The synonym dictionary is limited to common professional terms
- The optimization process maintains the original structure of the resume
- Word count increase is limited to 30 words
- The application works best with well-formatted PDF resumes

## Contributing

Feel free to submit issues and enhancement requests! 
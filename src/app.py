import streamlit as st
import os
import tempfile
import PyPDF2
import docx2txt
from werkzeug.utils import secure_filename
from typing import Dict, List, Any, Optional
import json
import re

# Import components from our modules
from src.preprocessing.parser import ResumeParser, JobParser
from src.preprocessing.cleaner import TextCleaner
from src.nlp.keyword_extractor import KeywordExtractor
from src.nlp.skills_analyzer import SkillsAnalyzer
from src.nlp.semantic_analyzer import SemanticAnalyzer
from src.matching.similarity import SimilarityMatcher
from src.matching.gap_analyzer import GapAnalyzer
from src.optimization.recommender import Recommender
from src.optimization.enhancer import ContentEnhancer
from src.utils.constants import API_CONFIG

# Configuration
API_KEY = os.environ.get("LLM_API_KEY", API_CONFIG.get("api_key", ""))
USE_LLM = bool(API_KEY)

# Initialize components
text_cleaner = TextCleaner()
resume_parser = ResumeParser()
job_parser = JobParser()
keyword_extractor = KeywordExtractor(industry="software")
skills_analyzer = SkillsAnalyzer()
semantic_analyzer = SemanticAnalyzer()
similarity_calculator = SimilarityMatcher(semantic_analyzer=semantic_analyzer)
gap_analyzer = GapAnalyzer(
    skills_analyzer=skills_analyzer, similarity_calculator=similarity_calculator
)
resume_recommender = Recommender(
    gap_analyzer=gap_analyzer,
    skills_analyzer=skills_analyzer,
    use_llm=USE_LLM,
    api_key=API_KEY,
)
content_enhancer = ContentEnhancer(
    keyword_extractor=keyword_extractor,
    semantic_analyzer=semantic_analyzer,
    text_cleaner=text_cleaner,
    use_llm=USE_LLM,
    api_key=API_KEY,
)


def main():
    st.title("Resume Optimization System")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Input", "Analysis", "Optimization"]
    choice = st.sidebar.radio("Go to", pages)

    if choice == "Input":
        render_input_page()
    elif choice == "Analysis":
        render_analysis_page()
    elif choice == "Optimization":
        render_optimization_page()


def render_input_page():
    st.header("Upload Documents")

    # Resume upload
    st.subheader("Resume")
    resume_upload = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"])
    resume_text = st.text_area("Or paste your resume text here", height=200)

    # Job description upload
    st.subheader("Job Description")
    job_upload = st.file_uploader("Upload job description", type=["pdf", "docx", "txt"])
    job_text = st.text_area("Or paste job description here", height=200)

    # Process button
    if st.button("Analyze Match"):
        # Check if either text or file is provided for both resume and job
        if (resume_upload or resume_text) and (job_upload or job_text):
            # Process resume
            if resume_upload:
                resume_content = extract_text_from_file(resume_upload)
            else:
                resume_content = resume_text

            # Process job description
            if job_upload:
                job_content = extract_text_from_file(job_upload)
            else:
                job_content = job_text

            # Store in session state
            st.session_state.resume_text = resume_content
            st.session_state.job_text = job_content

            # Run analysis
            results = analyze_resume_job_match(resume_content, job_content)

            # Store results in session state
            st.session_state.analysis_results = results

            # Navigate to analysis page
            st.experimental_rerun()
        else:
            st.error("Please provide both a resume and job description.")


def render_analysis_page():
    st.header("Resume Analysis Results")

    if not hasattr(st.session_state, "analysis_results"):
        st.warning("No analysis results available. Please upload documents first.")
        return

    results = st.session_state.analysis_results

    # Display match score
    st.subheader("Match Score")
    match_score = results.get("match_score", 0)
    st.progress(match_score / 100)
    st.metric("Match Score", f"{match_score:.1f}%")

    # Display gap analysis
    st.subheader("Gap Analysis")
    gap_analysis = results.get("gap_analysis", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        skills_match = gap_analysis.get("skills_match", 0)
        st.metric("Skills Match", f"{skills_match:.1f}%")
        st.progress(skills_match / 100)

    with col2:
        experience_match = gap_analysis.get("experience_match", 0)
        st.metric("Experience Match", f"{experience_match:.1f}%")
        st.progress(experience_match / 100)

    with col3:
        education_match = gap_analysis.get("education_match", 0)
        st.metric("Education Match", f"{education_match:.1f}%")
        st.progress(education_match / 100)

    # Display recommendations
    st.subheader("Key Recommendations")
    recommendations = results.get("recommendations", [])

    if recommendations:
        for i, rec in enumerate(recommendations[:5]):  # Show top 5 recommendations
            with st.expander(f"{i+1}. {rec.get('title', 'Recommendation')}"):
                st.write(rec.get("description", ""))
    else:
        st.info("No specific recommendations found.")

    # Display missing keywords
    st.subheader("Missing Keywords")
    missing_keywords = results.get("missing_keywords", [])

    if missing_keywords:
        keyword_cols = st.columns(3)
        for i, keyword in enumerate(missing_keywords):
            col_idx = i % 3
            # Handle both string and object formats for keywords
            if isinstance(keyword, dict):
                keyword_text = keyword.get("text", "")
            else:
                keyword_text = keyword
            keyword_cols[col_idx].markdown(f"- {keyword_text}")
    else:
        st.info("No missing keywords found.")

    # Enhance button
    if st.button("Enhance Resume"):
        if not hasattr(st.session_state, "resume_text") or not hasattr(
            st.session_state, "job_text"
        ):
            st.error(
                "Resume or job description not found. Please go back to the input page."
            )
            return

        # Run enhancement
        enhancement_results = enhance_resume(
            st.session_state.resume_text, st.session_state.job_text
        )

        # Store enhancement results
        st.session_state.enhancement_results = enhancement_results

        # Navigate to optimization page
        st.experimental_rerun()


def render_optimization_page():
    st.header("Resume Optimization Results")

    if not hasattr(st.session_state, "enhancement_results"):
        st.warning(
            "No enhancement results available. Please analyze your resume first."
        )
        return

    results = st.session_state.enhancement_results

    # Display enhanced match score
    st.subheader("Enhanced Match Score")
    enhanced_score = results.get("enhanced_score", 0)
    original_score = st.session_state.analysis_results.get("match_score", 0)
    improvement = enhanced_score - original_score

    col1, col2 = st.columns(2)
    with col1:
        st.progress(enhanced_score / 100)
        st.metric("Enhanced Score", f"{enhanced_score:.1f}%", f"+{improvement:.1f}%")
    with col2:
        st.write("Optimization Summary")
        summary = results.get("optimization_summary", {})
        for key, value in summary.items():
            st.write(f"- {key.replace('_', ' ').title()}: {value}")

    # Display enhanced bullet points
    st.subheader("Enhanced Bullet Points")
    enhanced_bullets = results.get("enhanced_bullets", [])

    if enhanced_bullets:
        for i, bullet in enumerate(enhanced_bullets):
            with st.expander(f"Bullet Point {i+1}"):
                st.markdown("**Original:**")
                st.write(bullet.get("original", ""))
                st.markdown("**Enhanced:**")
                st.write(bullet.get("enhanced", ""))

                if bullet.get("matched_keywords"):
                    st.markdown("**Keywords:**")
                    st.write(", ".join(bullet.get("matched_keywords", [])))
    else:
        st.info("No bullet points were enhanced.")

    # Display ATS tips
    st.subheader("ATS Optimization Tips")
    ats_tips = results.get("ats_tips", [])

    if ats_tips:
        for i, tip in enumerate(ats_tips):
            with st.expander(f"{tip.get('title', f'Tip {i+1}')}"):
                st.write(tip.get("description", ""))
    else:
        st.info("No ATS optimization tips available.")

    # Display enhanced resume
    st.subheader("Enhanced Resume")
    enhanced_resume = results.get("enhanced_resume", "")

    if enhanced_resume:
        # Add download button
        st.download_button(
            label="Download Enhanced Resume",
            data=enhanced_resume,
            file_name="enhanced_resume.txt",
            mime="text/plain",
        )

        # Show the enhanced resume
        with st.expander("View Enhanced Resume", expanded=True):
            st.text_area("", value=enhanced_resume, height=300, disabled=True)
    else:
        st.info("No enhanced resume available.")


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT files"""
    filename = uploaded_file.name
    file_ext = os.path.splitext(filename)[1].lower()

    if file_ext == ".pdf":
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Extract text from PDF
        text = ""
        with open(temp_file_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

        # Clean up
        os.unlink(temp_file_path)
        return text

    elif file_ext == ".docx":
        # Extract text from DOCX
        text = docx2txt.process(uploaded_file)
        return text

    elif file_ext == ".txt":
        # Read text file
        return uploaded_file.getvalue().decode("utf-8")

    else:
        raise ValueError(f"Unsupported file format: {file_ext}")


def analyze_resume_job_match(resume_text: str, job_text: str) -> Dict[str, Any]:
    """Analyze the match between a resume and job description"""
    try:
        # Parse and clean resume
        parsed_resume = resume_parser.parse(resume_text)
        cleaned_resume = {}
        for section, content in parsed_resume.get("sections", {}).items():
            cleaned_resume[section] = text_cleaner.clean_text(content)

        # Parse and clean job description
        parsed_job = job_parser.parse(job_text)
        cleaned_job = {}
        for section, content in parsed_job.get("sections", {}).items():
            cleaned_job[section] = text_cleaner.clean_text(content)

        # Extract keywords from job description
        job_keywords_tuples = keyword_extractor.extract_keywords(job_text, top_n=20)
        job_keywords = [
            {"text": kw, "score": score} for kw, score in job_keywords_tuples
        ]

        # Perform gap analysis
        gap_analysis = gap_analyzer.get_comprehensive_gap_analysis(
            {"sections": cleaned_resume},
            {
                "sections": cleaned_job,
                "metadata": {"industry": "software"},
            },  # Default to software industry
        )

        # Get recommendations
        recommendations = resume_recommender.get_context_aware_suggestions(
            {"sections": cleaned_resume},
            {
                "sections": cleaned_job,
                "metadata": {"title": parsed_job.get("title", "")},
            },
        )

        # Extract missing keywords from gap analysis and convert to proper format
        missing_skills = gap_analysis.get("skills_gap_analysis", {}).get(
            "missing_skills", []
        )
        # Convert string skills to dictionary format with 'text' key
        missing_keywords = [{"text": skill} for skill in missing_skills]

        # If we have very few missing keywords, extract more from job description
        if len(missing_keywords) <= 1:
            # Extract the most important keywords from the job
            resume_text_lower = resume_text.lower()
            additional_keywords = []
            for kw_dict in job_keywords:
                keyword = kw_dict["text"]
                # Only add keyword if it's not in the resume
                if keyword.lower() not in resume_text_lower:
                    additional_keywords.append({"text": keyword})

                # Limit to 10 missing keywords total
                if len(missing_keywords) + len(additional_keywords) >= 10:
                    break

            # Add the additional keywords to our missing keywords
            missing_keywords.extend(additional_keywords)
            # Remove duplicates
            unique_keywords = []
            seen = set()
            for kw in missing_keywords:
                if kw["text"].lower() not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw["text"].lower())
            missing_keywords = unique_keywords[:10]  # Limit to top 10 keywords

        # Calculate match score (weighted average of skills, experience, and education)
        match_score = gap_analysis.get("overall_match_score", 0) * 100

        # Extract gap metrics
        skills_match = gap_analysis.get("skills_gap_analysis", {}).get(
            "match_percentage", 0
        )

        experience_match = 0
        if "experience_gap_analysis" in gap_analysis:
            exp_gap = gap_analysis["experience_gap_analysis"]
            exp_compatibility = exp_gap.get("level_compatibility", 0) * 100
            exp_years_gap = exp_gap.get("experience_gap_years", 0)
            # Convert experience gap to match percentage (100% - gap%)
            max_exp_gap = 5  # Assume 5 years is maximum gap
            exp_years_match = 100 - min(100, (exp_years_gap / max_exp_gap) * 100)
            experience_match = (exp_compatibility + exp_years_match) / 2

        education_match = 0
        if "education_match_analysis" in gap_analysis:
            edu_match = gap_analysis["education_match_analysis"]
            education_match = edu_match.get("education_match_score", 0) * 100

        # Build response
        response = {
            "match_score": match_score,
            "missing_keywords": missing_keywords,
            "recommendations": recommendations[:10],  # Limit to top 10
            "gap_analysis": {
                "skills_match": skills_match,
                "experience_match": experience_match,
                "education_match": education_match,
            },
        }

        return response

    except Exception as e:
        print(f"Error in analyze_resume_job_match: {str(e)}")
        # Return a basic error response
        return {
            "error": f"Analysis failed: {str(e)}",
            "match_score": 0,
            "missing_keywords": [],
            "recommendations": [],
        }


def enhance_resume(resume_text: str, job_text: str) -> Dict[str, Any]:
    """Enhance the resume based on job description"""
    try:
        # Get analysis results first
        analysis_results = analyze_resume_job_match(resume_text, job_text)

        # Parse resume to get sections
        parsed_resume = resume_parser.parse(resume_text)

        # Parse job to get sections and keywords
        parsed_job = job_parser.parse(job_text)
        job_keywords_tuples = keyword_extractor.extract_keywords(job_text, top_n=30)
        job_keywords = [
            {"text": kw, "score": score} for kw, score in job_keywords_tuples
        ]

        # Extract bullet points from experience section
        experience_text = parsed_resume.get("sections", {}).get("experience", "")
        bullet_points = extract_bullet_points(experience_text)

        # Context for enhancements
        enhancement_context = {
            "job_title": parsed_job.get("title", ""),
            "job_description": job_text,
            "job_keywords": [kw["text"] for kw in job_keywords],
            "experience_level": parsed_job.get("experience_level", "mid"),
        }

        # Enhance bullet points
        enhanced_bullets = content_enhancer.enhance_bullet_points(
            bullet_points, enhancement_context
        )

        # Integrate keywords into resume sections
        enhanced_sections = {}
        sections_improved = 0

        for section, content in parsed_resume.get("sections", {}).items():
            if not content:
                enhanced_sections[section] = content
                continue

            # Skip sections that shouldn't have keywords integrated
            if section.lower() in ["contact", "header", "references"]:
                enhanced_sections[section] = content
                continue

            # Integrate relevant keywords for this section
            relevant_keywords = filter_keywords_for_section(section, job_keywords)

            if relevant_keywords:
                enhancement_result = content_enhancer.integrate_keywords(
                    content, relevant_keywords
                )
                enhanced_sections[section] = enhancement_result["enhanced_text"]

                if enhancement_result["keywords_added"]:
                    sections_improved += 1
            else:
                enhanced_sections[section] = content

        # Get ATS optimization tips
        ats_tips = resume_recommender.get_ats_optimization_tips(
            {
                "sections": parsed_resume.get("sections", {}),
                "metadata": {"format": "text"},
            }
        )

        # Build enhanced resume text
        enhanced_resume = ""
        for section, content in enhanced_sections.items():
            enhanced_resume += f"{section.upper()}\n{'-' * len(section)}\n{content}\n\n"

        # Calculate enhanced score (estimate improvement)
        original_score = analysis_results.get("match_score", 0)
        keyword_match_increase = (
            len([b for b in enhanced_bullets if b.get("matched_keywords")]) * 2
        )
        section_improvement = sections_improved * 5
        enhanced_score = min(
            100, original_score + keyword_match_increase + section_improvement
        )

        # Count enhanced bullets
        bullets_enhanced = len(
            [b for b in enhanced_bullets if b.get("enhancement_method") != "unchanged"]
        )

        # Build optimization summary
        optimization_summary = {
            "keyword_match_increase": keyword_match_increase,
            "bullets_enhanced": bullets_enhanced,
            "sections_improved": sections_improved,
            "score_improvement": enhanced_score - original_score,
        }

        # Build response
        response = {
            "enhanced_score": enhanced_score,
            "enhanced_resume": enhanced_resume,
            "enhanced_bullets": enhanced_bullets,
            "added_keywords": [
                kw
                for section in enhanced_sections.values()
                for kw in extract_added_keywords(section, resume_text)
            ],
            "ats_tips": ats_tips[:5],  # Limit to top 5
            "optimization_summary": optimization_summary,
        }

        return response

    except Exception as e:
        print(f"Error in enhance_resume: {str(e)}")
        # Return a basic error response
        return {
            "error": f"Enhancement failed: {str(e)}",
            "enhanced_score": 0,
            "enhanced_resume": resume_text,
            "enhanced_bullets": [],
            "added_keywords": [],
            "ats_tips": [],
        }


def extract_bullet_points(text: str) -> List[str]:
    """Extract bullet points from text"""
    # Split by common bullet point markers and newlines
    bullet_patterns = [
        r"•\s*([^\n•]+)",
        r"-\s*([^\n-]+)",
        r"\*\s*([^\n\*]+)",
        r"\d+\.\s*([^\n]+)",
    ]

    bullets = []

    for pattern in bullet_patterns:
        matches = re.findall(pattern, text)
        bullets.extend([match.strip() for match in matches if match.strip()])

    # If no bullets found using patterns, try splitting by newlines
    if not bullets:
        lines = text.split("\n")
        bullets = [line.strip() for line in lines if line.strip()]

    return bullets


def filter_keywords_for_section(
    section: str, keywords: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Filter keywords based on relevance to the resume section"""
    section = section.lower()

    # Define section-specific keyword filters
    section_keywords = {
        "summary": [
            "experience",
            "professional",
            "expertise",
            "background",
            "skilled",
            "years",
        ],
        "experience": [
            "managed",
            "developed",
            "created",
            "led",
            "implemented",
            "designed",
            "built",
        ],
        "skills": [
            "proficient",
            "expert",
            "knowledge",
            "advanced",
            "familiar",
            "skilled",
        ],
        "education": [
            "degree",
            "university",
            "college",
            "bachelor",
            "master",
            "phd",
            "certification",
        ],
        "projects": [
            "developed",
            "created",
            "built",
            "designed",
            "implemented",
            "collaborated",
        ],
    }

    # If section not in our mapping, return all keywords
    if section not in section_keywords:
        return keywords

    # Filter keywords based on section relevance
    relevant_terms = section_keywords[section]
    filtered_keywords = []

    for kw in keywords:
        # Check if any relevant term is in the keyword
        if any(term.lower() in kw["text"].lower() for term in relevant_terms):
            filtered_keywords.append(kw)
        # Always include high-score keywords
        elif kw.get("score", 0) > 0.8:
            filtered_keywords.append(kw)

    # If we filtered out too many, return top keywords from original list
    if len(filtered_keywords) < 5 and len(keywords) >= 5:
        return sorted(keywords, key=lambda x: x.get("score", 0), reverse=True)[:10]

    return filtered_keywords


def extract_added_keywords(enhanced_text: str, original_text: str) -> List[str]:
    """Extract keywords that were added to the enhanced text"""
    # Simple implementation - identify words in enhanced text not in original
    original_words = set(re.findall(r"\b\w+\b", original_text.lower()))
    enhanced_words = set(re.findall(r"\b\w+\b", enhanced_text.lower()))

    # Find words in enhanced that are not in original
    added_words = enhanced_words - original_words

    # Filter out short words and common words
    added_words = [word for word in added_words if len(word) > 3]

    return list(added_words)[:15]  # Limit to top 15


# API endpoints for Flask
def api_analyze():
    """Handle /api/analyze POST request"""
    try:
        # Get form data
        resume_text = request.form.get("resume_text", "")
        job_description = request.form.get("job_description", "")

        if not resume_text or not job_description:
            return (
                jsonify({"error": "Both resume and job description are required"}),
                400,
            )

        # Perform analysis
        results = analyze_resume_job_match(resume_text, job_description)

        # Return analysis results
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def api_enhance():
    """Handle /api/enhance POST request"""
    try:
        # Get form data
        resume_text = request.form.get("resume_text", "")
        job_description = request.form.get("job_description", "")

        if not resume_text or not job_description:
            return (
                jsonify({"error": "Both resume and job description are required"}),
                400,
            )

        # Perform enhancement
        results = enhance_resume(resume_text, job_description)

        # Return enhancement results
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Register API endpoints if using Flask
try:
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    # Register API endpoints
    app.route("/api/analyze", methods=["POST"])(api_analyze)
    app.route("/api/enhance", methods=["POST"])(api_enhance)

    # Serve static files
    @app.route("/")
    def index():
        return app.send_static_file("index.html")

    @app.route("/<path:path>")
    def static_files(path):
        return app.send_static_file(path)

except ImportError:
    # If Flask is not available, assume we're using Streamlit
    if __name__ == "__main__":
        main()
else:
    # If using Flask, define main entry point
    if __name__ == "__main__":
        # Check if running as Streamlit app
        if os.environ.get("STREAMLIT_RUN"):
            main()
        else:
            # Run Flask app
            app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

"""
Main Application
Streamlit interface for resume optimization system.
"""

import streamlit as st
import plotly.graph_objects as go
from src.app import extract_text_from_file
from src.preprocessing.parser import ResumeParser, JobParser
from src.preprocessing.cleaner import TextCleaner
from src.nlp.keyword_extractor import KeywordExtractor
from src.nlp.skills_analyzer import SkillsAnalyzer
from src.matching.similarity import SimilarityMatcher
from src.matching.gap_analyzer import GapAnalyzer
from src.optimization.recommender import Recommender
from src.optimization.enhancer import ContentEnhancer


def create_radar_chart(scores: dict):
    categories = list(scores.keys())
    values = list(scores.values())

    fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill="toself"))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False
    )

    return fig


def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")

    st.title("🚀 Resume Optimization System")
    st.markdown(
        """
    Optimize your resume for job descriptions using advanced NLP techniques.
    Upload your resume and paste the job description to get started.
    """
    )

    # Initialize components
    resume_parser = ResumeParser()
    job_parser = JobParser()
    matcher = SimilarityMatcher()
    gap_analyzer = GapAnalyzer()
    recommender = Recommender()
    enhancer = ContentEnhancer()

    # File uploader for resume
    resume_file = st.file_uploader(
        "Upload your resume (PDF or DOCX)", type=["pdf", "docx"]
    )

    # Text area for job description
    job_description = st.text_area("Paste the job description here", height=200)

    if resume_file and job_description:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                # Process resume
                resume_text = extract_text_from_file(resume_file)
                # Then extract sections from the text
                resume_data = resume_parser.parse(resume_text)
                # Process job description
                job_data = job_parser.parse(job_description)

                # Calculate matches and gaps
                similarity_result = matcher.get_document_similarities(
                    resume_data,  # Pass the full resume_data
                    job_data,  # Pass the full job_data
                )

                # Extract data for the radar chart
                section_scores = {}
                for section, similarity in similarity_result[
                    "section_similarities"
                ].items():
                    # Extract the 'combined' score from each section's similarity data
                    if isinstance(similarity, dict) and "combined" in similarity:
                        section_scores[section.capitalize()] = similarity["combined"]

                # Add overall similarity to the chart
                section_scores["Overall"] = similarity_result["overall_similarity"]

                # Get comprehensive gap analysis
                gap_analysis = gap_analyzer.get_comprehensive_gap_analysis(
                    resume_data, job_data
                )

                # Generate recommendations
                recommendations = recommender.get_context_aware_suggestions(
                    resume_data, job_data
                )

                st.success("Analysis complete!")

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Match Analysis")
                    fig = create_radar_chart(section_scores)
                    st.plotly_chart(fig)

                with col2:
                    st.subheader("Skills Gap Analysis")
                    # Extract skills data from comprehensive analysis
                    skills_analysis = gap_analysis.get("skills_gap_analysis", {})

                    # Show match percentage
                    match_percentage = skills_analysis.get("match_percentage", 0)
                    st.progress(match_percentage / 100)
                    st.write(f"Skills Match: {match_percentage:.1f}%")

                    # Show matching skills
                    st.write("**Matching Skills:**")
                    for skill in skills_analysis.get("matching_skills", [])[
                        :5
                    ]:  # Show top 5
                        st.write(f"✅ {skill}")

                    # Show missing skills
                    st.write("**Missing Skills:**")
                    for skill in skills_analysis.get("missing_skills", [])[
                        :5
                    ]:  # Show top 5
                        st.write(f"❌ {skill}")

                st.subheader("Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec.get('description', '')}")
                    if "action_items" in rec:
                        for action in rec["action_items"]:
                            st.write(f"   • {action}")


if __name__ == "__main__":
    main()

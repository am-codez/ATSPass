"""
Main Application
Streamlit interface for resume optimization system.
"""

import streamlit as st
import plotly.graph_objects as go
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
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )
    
    return fig

def main():
    st.set_page_config(page_title="Resume Optimizer", layout="wide")
    
    st.title("🚀 Resume Optimization System")
    st.markdown("""
    Optimize your resume for job descriptions using advanced NLP techniques.
    Upload your resume and paste the job description to get started.
    """)
    
    # Initialize components
    resume_parser = ResumeParser()
    job_parser = JobParser()
    text_cleaner = TextCleaner()
    skills_analyzer = SkillsAnalyzer()
    matcher = SimilarityMatcher()
    gap_analyzer = GapAnalyzer()
    recommender = Recommender()
    enhancer = ContentEnhancer()
    
    # File uploader for resume
    resume_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=['pdf', 'docx'])
    
    # Text area for job description
    job_description = st.text_area("Paste the job description here", height=200)
    
    if resume_file and job_description:
        if st.button("Analyze Resume"):
            with st.spinner("Analyzing your resume..."):
                # Process resume
                resume_text = resume_parser.parse_pdf(resume_file) if resume_file.type == 'application/pdf' \
                            else resume_parser.parse_docx(resume_file)
                resume_data = resume_parser.extract_sections(resume_text)
                
                # Process job description
                job_data = job_parser.extract_sections(job_description)
                
                # Calculate matches and gaps
                similarity_scores = matcher.calculate_overall_similarity(resume_data, job_data)
                gaps = gap_analyzer.calculate_gap_scores(resume_data, job_data)
                
                # Generate recommendations
                recommendations = recommender.generate_recommendations(resume_data, job_data)
                
                st.success("Analysis complete!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Match Analysis")
                    fig = create_radar_chart(similarity_scores)
                    st.plotly_chart(fig)
                
                with col2:
                    st.subheader("Skills Gap Analysis")
                    for skill, score in gaps.items():
                        st.write(f"{'✅' if score > 0.7 else '❌'} {skill}: {score:.0%}")
                
                st.subheader("Recommendations")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec['description']}")
                    if 'action_items' in rec:
                        for action in rec['action_items']:
                            st.write(f"   • {action}")

if __name__ == "__main__":
    main()

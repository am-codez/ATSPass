"""
Resume formatting utilities.
"""

from typing import List

def format_resume_section(section_name: str, content: List[str]) -> str:
    """Format a resume section with proper spacing and structure."""
    if not content:
        return ""
    
    # Format section header
    formatted = f"\n{section_name}\n"
    
    # Add content with proper indentation
    for line in content:
        if line.strip():
            # If line starts with a bullet point or is a job title, don't indent
            if line.strip().startswith(('•', '-', '*')) or any(line.strip().startswith(title) for title in ['DevOps Engineer', 'Site Reliability Engineer', 'Cloud Engineer', 'Software Engineer']):
                formatted += f"\n{line.strip()}"
            else:
                formatted += f"\n    {line.strip()}"
    
    return formatted 
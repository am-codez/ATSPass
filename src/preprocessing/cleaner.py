"""
Text Cleaning Module
Handles text preprocessing and normalization.
"""

import re
import nltk
from typing import List, Tuple
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        
        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Add resume-specific stop words to keep
        self.keep_words = {
            # Technical Skills
            'python', 'java', 'sql', 'aws', 'docker', 'kubernetes', 'react', 'angular', 'vue',
            'javascript', 'typescript', 'node', 'django', 'flask', 'spring', 'hibernate',
            'mongodb', 'postgresql', 'mysql', 'redis', 'elasticsearch', 'kafka', 'spark',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib',
            
            # Cloud & DevOps
            'azure', 'gcp', 'jenkins', 'git', 'github', 'gitlab', 'ci', 'cd', 'terraform',
            'ansible', 'prometheus', 'grafana', 'elk', 'splunk', 'jira', 'confluence',
            
            # Action Verbs
            'led', 'managed', 'developed', 'created', 'designed', 'implemented', 'architected',
            'optimized', 'improved', 'reduced', 'increased', 'achieved', 'delivered', 'launched',
            'maintained', 'debugged', 'tested', 'deployed', 'monitored', 'analyzed', 'researched',
            
            # Education Terms
            'bs', 'ms', 'phd', 'bachelor', 'master', 'doctorate', 'degree', 'diploma',
            'certification', 'certified', 'accredited', 'gpa', 'cum', 'laude',
            
            # Professional Terms
            'senior', 'junior', 'lead', 'principal', 'architect', 'engineer', 'developer',
            'consultant', 'analyst', 'manager', 'director', 'head', 'chief', 'president',
            
            # Industry Terms
            'agile', 'scrum', 'waterfall', 'kanban', 'sprint', 'backlog', 'standup',
            'retrospective', 'roadmap', 'milestone', 'deadline', 'budget', 'stakeholder',
            
            # Soft Skills
            'leadership', 'communication', 'collaboration', 'problem-solving', 'analytical',
            'strategic', 'innovative', 'creative', 'detail-oriented', 'self-motivated',
            
            # Project Terms
            'project', 'team', 'client', 'stakeholder', 'requirement', 'specification',
            'documentation', 'review', 'feedback', 'presentation', 'report', 'analysis'
        }

    def clean_text(self, text: str) -> str:
        """Main cleaning pipeline"""
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)
        return text

    def remove_special_characters(self, text: str) -> str:
        """Remove special characters while preserving important symbols"""
        # Keep emails intact
        emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
        
        # Keep URLs intact
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)
        
        # Replace emails and URLs with placeholders
        for i, email in enumerate(emails):
            text = text.replace(email, f'EMAIL_{i}')
        
        for i, url in enumerate(urls):
            text = text.replace(url, f'URL_{i}')
        
        # Remove special characters except those important for resumes
        # Keep: alphanumeric, spaces, @.-+% (for emails & dates)
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s@.+\-,%]', ' ', text)
        
        # Restore emails and URLs
        for i, email in enumerate(emails):
            cleaned_text = cleaned_text.replace(f'EMAIL_{i}', email)
        
        for i, url in enumerate(urls):
            cleaned_text = cleaned_text.replace(f'URL_{i}', url)
        
        return cleaned_text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks"""
        # Replace various line break types with standard newline
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Handle bullet points - ensure space after bullet
        text = re.sub(r'(•|\*|\-|–)(?=\S)', r'\1 ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Remove spaces at start/end of lines
        text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
        
        # Remove extra blank lines (more than one consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()

    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Use NLTK's tokenizer
        # Handle special cases like dates, numbers
        return word_tokenize(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stop words while preserving important resume terms"""
        # Keep resume-specific terms
        # Remove other stop words
        return [token for token in tokens if token not in self.stop_words or token in self.keep_words]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens while preserving certain forms"""
        # Lemmatize most words
        # Preserve certain forms (e.g., action verbs)
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def preserve_bullet_points(self, text: str) -> str:
        """Preserve bullet points while cleaning"""
        # Replace different bullet point characters with standard ones
        bullet_patterns = [r'•', r'○', r'●', r'■', r'□', r'▪', r'▫']
        for pattern in bullet_patterns:
            text = re.sub(pattern, '•', text)
        return text
    
    def preserve_dates(self, text: str) -> Tuple[str, List[str]]:
        """Preserve date formats while cleaning"""
        # Store dates temporarily
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'\d{4}-\d{1,2}-\d{1,2}',    # YYYY-MM-DD
            r'[A-Za-z]+ \d{4}',          # Month YYYY
            r'\d{4}',                    # YYYY
            r'[A-Za-z]+ \d{1,2},? \d{4}' # Month DD, YYYY
        ]
        
        # Find and store all dates
        for pattern in date_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                dates.append(match.group())
        
        # Replace dates with placeholders
        for i, date in enumerate(dates):
            text = text.replace(date, f'DATE_{i}')
        
        return text, dates

    def preserve_technical_terms(self, text: str) -> Tuple[str, List[str]]:
        """Preserve technical terms and versions"""
        # Store technical terms temporarily
        tech_terms = []
        
        # Common technical term patterns
        tech_patterns = [
            # Programming Languages
            r'Python \d\.\d+',           # Python versions
            r'Java \d+',                # Java versions
            r'JavaScript ES\d+',        # JavaScript versions
            r'TypeScript \d+\.\d+',     # TypeScript versions
            r'C\+\+ \d+',               # C++ versions
            r'C# \d+\.\d+',             # C# versions
            r'Ruby \d+\.\d+',           # Ruby versions
            r'PHP \d+\.\d+',            # PHP versions
            r'Go \d+\.\d+',             # Go versions
            r'Rust \d+\.\d+',           # Rust versions
            r'Swift \d+\.\d+',          # Swift versions
            r'Kotlin \d+\.\d+',         # Kotlin versions
            r'Scala \d+\.\d+',          # Scala versions
            r'R \d+\.\d+',              # R versions
            
            # Databases
            r'SQL Server \d{4}',        # SQL Server versions
            r'MySQL \d+\.\d+',          # MySQL versions
            r'PostgreSQL \d+\.\d+',     # PostgreSQL versions
            r'Oracle Database \d+c?',   # Oracle versions
            r'MongoDB \d+\.\d+',        # MongoDB versions
            r'Redis \d+\.\d+',          # Redis versions
            r'Cassandra \d+\.\d+',      # Cassandra versions
            r'Neo4j \d+\.\d+',          # Neo4j versions
            r'DynamoDB',                # DynamoDB
            r'Elasticsearch \d+\.\d+',  # Elasticsearch versions
            
            # Cloud Services
            r'AWS [A-Za-z]+',           # AWS services
            r'Amazon S3|EC2|RDS|Lambda|SQS|SNS|ECS|EKS|CloudFront',  # Common AWS services
            r'Azure [A-Za-z]+',         # Azure services
            r'Google Cloud [A-Za-z]+',  # GCP services
            r'GCP [A-Za-z]+',           # GCP abbreviated
            r'Firebase',                # Firebase
            r'Heroku',                  # Heroku
            r'DigitalOcean',            # DigitalOcean
            
            # Web Frameworks
            r'React \d+\.\d+',          # React versions
            r'Angular \d+',             # Angular versions
            r'Vue\.js \d+\.\d+',        # Vue.js versions
            r'Next\.js \d+\.\d+',       # Next.js versions
            r'Nuxt\.js \d+\.\d+',       # Nuxt.js versions
            r'Express\.js \d+\.\d+',    # Express.js versions
            r'Django \d+\.\d+',         # Django versions
            r'Flask \d+\.\d+',          # Flask versions
            r'Spring Boot \d+\.\d+',    # Spring Boot versions
            r'Laravel \d+\.\d+',        # Laravel versions
            r'ASP\.NET Core \d+\.\d+',  # ASP.NET Core versions
            r'Ruby on Rails \d+\.\d+',  # Rails versions
            
            # Backend Technologies
            r'Node\.js \d+\.\d+',       # Node.js versions
            r'Deno \d+\.\d+',           # Deno versions
            r'GraphQL',                 # GraphQL
            r'RESTful API',             # REST APIs
            r'gRPC',                    # gRPC
            r'WebSockets',              # WebSockets
            r'Socket\.IO',              # Socket.IO
            
            # DevOps & Tooling
            r'Docker \d+\.\d+',         # Docker versions
            r'Kubernetes \d+\.\d+',     # Kubernetes versions
            r'Git \d+\.\d+',            # Git versions
            r'Jenkins \d+\.\d+',        # Jenkins versions
            r'Travis CI',               # Travis CI
            r'CircleCI',                # CircleCI
            r'GitHub Actions',          # GitHub Actions
            r'GitLab CI/CD',            # GitLab CI/CD
            r'Terraform \d+\.\d+',      # Terraform versions
            r'Ansible \d+\.\d+',        # Ansible versions
            r'Puppet \d+\.\d+',         # Puppet versions
            r'Chef \d+\.\d+',           # Chef versions
            r'Prometheus',              # Prometheus
            r'Grafana',                 # Grafana
            r'ELK Stack',               # ELK Stack
            
            # AI & Data Science
            r'TensorFlow \d+\.\d+',     # TensorFlow versions
            r'PyTorch \d+\.\d+',        # PyTorch versions
            r'Keras \d+\.\d+',          # Keras versions
            r'Scikit-learn \d+\.\d+',   # Scikit-learn versions
            r'Pandas \d+\.\d+',         # Pandas versions
            r'NumPy \d+\.\d+',          # NumPy versions
            r'Matplotlib \d+\.\d+',     # Matplotlib versions
            r'Seaborn \d+\.\d+',        # Seaborn versions
            r'NLTK \d+\.\d+',           # NLTK versions
            r'spaCy \d+\.\d+',          # spaCy versions
            r'Hugging Face',            # Hugging Face
            r'Transformers \d+\.\d+',   # Transformers versions
            r'BERT|GPT|RoBERTa|XLNet',  # Common AI models
            r'Hadoop \d+\.\d+',         # Hadoop versions
            r'Spark \d+\.\d+',          # Spark versions
            r'Kafka \d+\.\d+',          # Kafka versions
            r'Airflow \d+\.\d+',        # Airflow versions
            r'Databricks',              # Databricks
            
            # Mobile Development
            r'Android \d+',             # Android versions
            r'iOS \d+',                 # iOS versions
            r'React Native \d+\.\d+',   # React Native versions
            r'Flutter \d+\.\d+',        # Flutter versions
            r'Xamarin \d+\.\d+',        # Xamarin versions
            r'Cordova \d+\.\d+',        # Cordova versions
            r'Ionic \d+\.\d+',          # Ionic versions
            
            # Testing & QA
            r'Jest \d+\.\d+',           # Jest versions
            r'Mocha \d+\.\d+',          # Mocha versions
            r'Cypress \d+\.\d+',        # Cypress versions
            r'Selenium \d+\.\d+',       # Selenium versions
            r'JUnit \d+\.\d+',          # JUnit versions
            r'TestNG \d+\.\d+',         # TestNG versions
            r'PyTest \d+\.\d+',         # PyTest versions
            
            # Project Management & Collaboration
            r'Jira',                    # Jira
            r'Confluence',              # Confluence
            r'Trello',                  # Trello
            r'Asana',                   # Asana
            r'Slack',                   # Slack
            r'Microsoft Teams',         # Microsoft Teams
            r'Agile',                   # Agile
            r'Scrum',                   # Scrum
            r'Kanban',                  # Kanban
            r'Waterfall',               # Waterfall
        ]
        
        # Find and store all technical terms
        for pattern in tech_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                tech_terms.append(match.group())
        
        # Replace technical terms with placeholders
        for i, term in enumerate(tech_terms):
            text = text.replace(term, f'TECH_{i}')
        
        return text, tech_terms

    def clean_and_tokenize(self, text: str) -> List[str]:
        """Complete cleaning and tokenization pipeline"""
        # Preserve dates and technical terms
        text, dates = self.preserve_dates(text)
        text, tech_terms = self.preserve_technical_terms(text)
        
        # Clean text
        text = self.clean_text(text)
        
        # Preserve bullet points
        text = self.preserve_bullet_points(text)
        
        # Tokenize
        tokens = self.tokenize_text(text)
        
        # Remove stopwords while preserving important terms
        tokens = self.remove_stopwords(tokens)
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Restore dates and technical terms
        for i, date in enumerate(dates):
            tokens = [token.replace(f'DATE_{i}', date) for token in tokens]
        
        for i, term in enumerate(tech_terms):
            tokens = [token.replace(f'TECH_{i}', term) for token in tokens]
        
        return tokens
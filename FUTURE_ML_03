

import re
import math
import string
from collections import Counter

# ─────────────────────────────────────────────
# 1. JOB ROLES & DESCRIPTIONS
# ─────────────────────────────────────────────

JOB_ROLES = {
    "data_scientist": {
        "title": "Data Scientist",
        "required_skills": [
            "python", "machine learning", "deep learning", "tensorflow", "pytorch",
            "pandas", "numpy", "scikit-learn", "statistics", "sql", "data visualization",
            "nlp", "neural networks", "feature engineering", "model evaluation"
        ],
        "nice_to_have": ["spark", "aws", "docker", "mlflow", "airflow", "tableau"],
        "experience_keywords": ["model", "trained", "deployed", "accuracy", "dataset", "research"],
        "min_experience_years": 2,
    },
    "frontend_developer": {
        "title": "Frontend Developer",
        "required_skills": [
            "javascript", "react", "html", "css", "typescript", "rest api",
            "git", "responsive design", "webpack", "node.js"
        ],
        "nice_to_have": ["vue", "angular", "graphql", "jest", "storybook", "figma", "tailwind"],
        "experience_keywords": ["built", "developed", "designed", "deployed", "ui", "ux"],
        "min_experience_years": 1,
    },
    "backend_developer": {
        "title": "Backend Developer",
        "required_skills": [
            "python", "java", "sql", "rest api", "docker", "git",
            "postgresql", "redis", "microservices", "linux"
        ],
        "nice_to_have": ["kubernetes", "aws", "kafka", "elasticsearch", "go", "rabbitmq"],
        "experience_keywords": ["built", "designed", "optimized", "scaled", "deployed", "api"],
        "min_experience_years": 2,
    },
    "devops_engineer": {
        "title": "DevOps Engineer",
        "required_skills": [
            "docker", "kubernetes", "ci/cd", "linux", "aws", "terraform",
            "git", "bash", "monitoring", "ansible"
        ],
        "nice_to_have": ["azure", "gcp", "prometheus", "grafana", "jenkins", "helm"],
        "experience_keywords": ["automated", "deployed", "pipeline", "infrastructure", "scaled"],
        "min_experience_years": 2,
    },
}

# ─────────────────────────────────────────────
# 2. SAMPLE RESUMES
# ─────────────────────────────────────────────

SAMPLE_RESUMES = [
    {
        "name": "Alice Chen",
        "text": """
        Alice Chen | alice@email.com | github.com/alicechen
        
        EXPERIENCE
        Senior Data Scientist, TechCorp (3 years)
        - Built and deployed machine learning models using Python, TensorFlow, PyTorch
        - Feature engineering on large datasets with Pandas and NumPy
        - NLP pipelines for text classification achieving 94% accuracy
        - Deep learning neural networks for image recognition
        - Model evaluation and performance monitoring with MLflow
        
        Data Analyst, DataInc (2 years)
        - SQL queries and data visualization with Tableau
        - Statistical analysis and A/B testing
        - Scikit-learn for predictive models
        
        SKILLS
        Python, Machine Learning, Deep Learning, TensorFlow, PyTorch, NLP, 
        SQL, Pandas, NumPy, Scikit-learn, Statistics, Feature Engineering,
        Data Visualization, Spark, AWS, Docker, MLflow
        
        EDUCATION
        MS Computer Science, Stanford University
        """
    },
    {
        "name": "Bob Martinez",
        "text": """
        Bob Martinez | bob@email.com
        
        EXPERIENCE
        Frontend Developer, StartupXYZ (2 years)
        - Built React components with TypeScript and modern JavaScript
        - Responsive design with CSS, HTML and Tailwind
        - REST API integration and state management
        - Webpack configuration and build optimization
        - Unit testing with Jest, component library with Storybook
        
        Junior Developer, WebShop (1 year)
        - HTML, CSS, JavaScript development
        - Git version control
        - Node.js backend integration
        
        SKILLS
        JavaScript, React, TypeScript, HTML, CSS, Node.js, Git,
        REST API, Webpack, Jest, Storybook, Tailwind, Figma, Vue
        
        EDUCATION
        BS Computer Science
        """
    },
    {
        "name": "Carol Singh",
        "text": """
        Carol Singh | carol@email.com
        
        EXPERIENCE
        ML Engineer, AIStartup (1 year)
        - Python development for data pipelines
        - Basic machine learning with Scikit-learn
        - SQL database management
        - Statistics and data analysis
        
        SKILLS
        Python, SQL, Statistics, Scikit-learn, Pandas, NumPy,
        Data Visualization, Git, Linux
        
        EDUCATION
        BS Mathematics
        """
    },
    {
        "name": "David Park",
        "text": """
        David Park | david@email.com
        
        EXPERIENCE
        DevOps Engineer, CloudCo (4 years)
        - Kubernetes and Docker container orchestration
        - CI/CD pipelines with Jenkins and GitHub Actions
        - AWS infrastructure with Terraform and Ansible
        - Linux server administration and Bash scripting
        - Monitoring with Prometheus and Grafana
        
        Systems Admin, OldCorp (2 years)
        - Linux administration
        - Automated deployments
        - Network and infrastructure management
        
        SKILLS
        Docker, Kubernetes, AWS, Terraform, CI/CD, Linux, Bash,
        Ansible, Git, Prometheus, Grafana, Jenkins, Helm, Azure
        
        EDUCATION
        BS Information Technology
        """
    },
    {
        "name": "Eva Liu",
        "text": """
        Eva Liu | eva@email.com
        
        EXPERIENCE
        Backend Developer, FinTech Ltd (3 years)
        - Python and Java microservices architecture
        - PostgreSQL and Redis database design
        - REST API development and Docker containerization
        - Linux server management and Git workflows
        - SQL optimization for high-traffic applications
        
        Software Engineer, SmallCo (1 year)
        - Python backend development
        - REST API integration
        - Microservices design
        
        SKILLS
        Python, Java, SQL, PostgreSQL, Redis, Docker, Linux, Git,
        REST API, Microservices, Kafka, Elasticsearch, AWS
        
        EDUCATION
        BS Software Engineering
        """
    },
]

# ─────────────────────────────────────────────
# 3. TEXT PREPROCESSING
# ─────────────────────────────────────────────

STOPWORDS = {
    "i","me","my","we","our","you","your","he","she","it","they","this","that",
    "the","a","an","and","or","but","in","on","at","to","for","of","with","is",
    "are","was","were","be","been","have","has","had","do","does","did","will",
    "would","can","could","should","may","might","not","no","so","if","as",
    "from","by","also","any","all","more","about","up","out","very","just"
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^\w\s/\.]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text: str) -> list:
    tokens = clean_text(text).split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

def extract_bigrams(tokens: list) -> list:
    return [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]


# ─────────────────────────────────────────────
# 4. SKILL EXTRACTION
# ─────────────────────────────────────────────

def extract_skills(resume_text: str, skill_list: list) -> list:
    """Find which skills from skill_list appear in the resume text."""
    text_lower = resume_text.lower()
    found = []
    for skill in skill_list:
        # match whole word / phrase
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


# ─────────────────────────────────────────────
# 5. EXPERIENCE YEAR EXTRACTION
# ─────────────────────────────────────────────

def extract_experience_years(resume_text: str) -> float:
    """Estimate total years of experience from patterns like '3 years', '(2 years)'."""
    patterns = [
        r'(\d+)\s+years?',
        r'\((\d+)\s+years?\)',
        r'(\d+)\+\s*years?',
    ]
    years = []
    for pat in patterns:
        matches = re.findall(pat, resume_text.lower())
        years.extend(int(m) for m in matches)
    if not years:
        return 0
    # sum role-level years, capped at 15
    return min(sum(years), 15)


# ─────────────────────────────────────────────
# 6. TF-IDF SIMILARITY SCORER
# ─────────────────────────────────────────────

def tfidf_cosine_similarity(text_a: str, text_b: str) -> float:
    """Compute cosine similarity using simple TF-IDF over two documents."""
    tokens_a = tokenize(text_a) + extract_bigrams(tokenize(text_a))
    tokens_b = tokenize(text_b) + extract_bigrams(tokenize(text_b))

    vocab = list(set(tokens_a) | set(tokens_b))
    if not vocab:
        return 0.0

    def tf(tokens, word):
        return tokens.count(word) / len(tokens) if tokens else 0

    def idf(word):
        docs_with = sum(1 for d in [tokens_a, tokens_b] if word in d)
        return math.log((2 + 1) / (docs_with + 1)) + 1  # smoothed

    def tfidf_vec(tokens):
        return [tf(tokens, w) * idf(w) for w in vocab]

    va, vb = tfidf_vec(tokens_a), tfidf_vec(tokens_b)
    dot = sum(a * b for a, b in zip(va, vb))
    mag_a = math.sqrt(sum(a**2 for a in va))
    mag_b = math.sqrt(sum(b**2 for b in vb))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ─────────────────────────────────────────────
# 7. SCORING ENGINE
# ─────────────────────────────────────────────

def score_resume(resume: dict, job_role: dict) -> dict:
    text = resume["text"]
    req  = job_role["required_skills"]
    nice = job_role["nice_to_have"]
    exp_kw = job_role["experience_keywords"]

    # A) Required skill match
    matched_req  = extract_skills(text, req)
    matched_nice = extract_skills(text, nice)
    skill_score  = len(matched_req) / len(req) if req else 0  # 0–1

    # B) Nice-to-have bonus (up to 0.1)
    nice_bonus = min(len(matched_nice) / max(len(nice), 1), 1.0) * 0.1

    # C) Experience score
    years = extract_experience_years(text)
    min_yr = job_role["min_experience_years"]
    exp_score = min(years / max(min_yr * 2, 4), 1.0)  # saturates at 2× minimum

    # D) TF-IDF similarity with job description (skill list as proxy)
    jd_text = " ".join(req + nice + exp_kw)
    tfidf_score = tfidf_cosine_similarity(text, jd_text)

    # E) Experience keyword density
    text_lower = text.lower()
    kw_hits = sum(1 for kw in exp_kw if kw in text_lower)
    kw_score = min(kw_hits / max(len(exp_kw), 1), 1.0)

    # Weighted total (out of 100)
    total = (
        skill_score  * 45 +
        exp_score    * 20 +
        tfidf_score  * 20 +
        kw_score     * 10 +
        nice_bonus   * 10 * 10   # normalise bonus back to contribution
    )
    total = min(round(total, 1), 100)

    missing = [s for s in req if s not in matched_req]

    return {
        "name": resume["name"],
        "total_score": total,
        "skill_score": round(skill_score * 100, 1),
        "experience_score": round(exp_score * 100, 1),
        "tfidf_score": round(tfidf_score * 100, 1),
        "years_detected": years,
        "matched_required": matched_req,
        "matched_nice": matched_nice,
        "missing_skills": missing,
        "fit_label": "Strong Fit" if total >= 70 else "Moderate Fit" if total >= 45 else "Weak Fit",
    }


# ─────────────────────────────────────────────
# 8. RANKING ENGINE
# ─────────────────────────────────────────────

def rank_resumes(resumes: list, role_key: str) -> list:
    job_role = JOB_ROLES[role_key]
    scored = [score_resume(r, job_role) for r in resumes]
    return sorted(scored, key=lambda x: x["total_score"], reverse=True)


# ─────────────────────────────────────────────
# 9. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    for role_key, role_info in JOB_ROLES.items():
        print(f"\n{'='*60}")
        print(f"  JOB ROLE: {role_info['title']}")
        print(f"{'='*60}")
        ranked = rank_resumes(SAMPLE_RESUMES, role_key)
        for i, r in enumerate(ranked, 1):
            print(f"\n#{i} {r['name']}  [{r['fit_label']}]  Score: {r['total_score']}/100")
            print(f"   Skills matched : {len(r['matched_required'])}/{len(role_info['required_skills'])} required")
            print(f"   Experience      : {r['years_detected']} years detected")
            print(f"   Missing skills  : {', '.join(r['missing_skills'][:5]) or 'None'}")

from flask import Blueprint, request, jsonify
import re
import json
import traceback
from collections import Counter
import numpy as np

cv_bp = Blueprint('cv', __name__)

class AdvancedCVAnalyzer:
    """Advanced CV Analyzer using NLP and ML techniques"""
    
    def __init__(self):
        self.skills_database = self._load_skills_database()
        self.job_titles_database = self._load_job_titles_database()
        
    def _load_skills_database(self):
        """Load comprehensive skills database"""
        return {
            'programming': [
                'Python', 'JavaScript', 'Java', 'C++', 'C#', 'PHP', 'Ruby', 'Go', 'Rust',
                'TypeScript', 'Kotlin', 'Swift', 'R', 'MATLAB', 'Scala', 'Perl'
            ],
            'web_development': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Express', 'Django', 'Flask',
                'Spring', 'Laravel', 'ASP.NET', 'HTML', 'CSS', 'SASS', 'Bootstrap'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Oracle', 'SQL Server',
                'SQLite', 'Cassandra', 'DynamoDB', 'Firebase'
            ],
            'cloud_platforms': [
                'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Terraform',
                'Jenkins', 'GitLab CI', 'GitHub Actions'
            ],
            'data_science': [
                'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Pandas',
                'NumPy', 'Scikit-learn', 'Jupyter', 'Tableau', 'Power BI'
            ],
            'soft_skills': [
                'Leadership', 'Communication', 'Problem Solving', 'Team Work',
                'Project Management', 'Critical Thinking', 'Creativity', 'Adaptability'
            ]
        }
    
    def _load_job_titles_database(self):
        """Load job titles database for classification"""
        return {
            'software_engineer': [
                'Software Engineer', 'Software Developer', 'Full Stack Developer',
                'Frontend Developer', 'Backend Developer', 'Web Developer'
            ],
            'data_scientist': [
                'Data Scientist', 'Data Analyst', 'Machine Learning Engineer',
                'AI Engineer', 'Research Scientist'
            ],
            'product_manager': [
                'Product Manager', 'Product Owner', 'Program Manager',
                'Project Manager', 'Scrum Master'
            ],
            'designer': [
                'UI Designer', 'UX Designer', 'Graphic Designer',
                'Product Designer', 'Visual Designer'
            ]
        }
    
    def extract_contact_info(self, text):
        """Extract contact information using regex"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        
        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)
        
        return {
            'emails': emails,
            'phones': [phone for phone in phones if len(re.sub(r'[^\d]', '', phone)) >= 10]
        }
    
    def extract_skills(self, text):
        """Extract skills using advanced NLP techniques"""
        text_lower = text.lower()
        found_skills = {}
        
        for category, skills in self.skills_database.items():
            found_skills[category] = []
            for skill in skills:
                if skill.lower() in text_lower:
                    # Calculate skill proficiency based on context
                    proficiency = self._calculate_skill_proficiency(text, skill)
                    found_skills[category].append({
                        'name': skill,
                        'proficiency': proficiency,
                        'mentions': text_lower.count(skill.lower())
                    })
        
        return found_skills
    
    def _calculate_skill_proficiency(self, text, skill):
        """Calculate skill proficiency based on context"""
        skill_lower = skill.lower()
        text_lower = text.lower()
        
        # Look for proficiency indicators
        expert_indicators = ['expert', 'advanced', 'senior', 'lead', 'architect']
        intermediate_indicators = ['intermediate', 'experienced', 'proficient']
        beginner_indicators = ['beginner', 'basic', 'learning', 'familiar']
        
        # Find skill context
        skill_positions = [i for i in range(len(text_lower)) if text_lower.startswith(skill_lower, i)]
        
        for pos in skill_positions:
            context_start = max(0, pos - 50)
            context_end = min(len(text_lower), pos + len(skill_lower) + 50)
            context = text_lower[context_start:context_end]
            
            if any(indicator in context for indicator in expert_indicators):
                return 'Expert'
            elif any(indicator in context for indicator in intermediate_indicators):
                return 'Intermediate'
            elif any(indicator in context for indicator in beginner_indicators):
                return 'Beginner'
        
        # Default to intermediate if no specific indicator found
        return 'Intermediate'
    
    def extract_experience(self, text):
        """Extract work experience information"""
        # Pattern for years of experience
        years_pattern = r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)'
        years_matches = re.findall(years_pattern, text.lower())
        
        total_years = 0
        if years_matches:
            total_years = max([int(year) for year in years_matches])
        
        # Extract job titles and companies
        job_sections = self._extract_job_sections(text)
        
        return {
            'total_years': total_years,
            'positions': job_sections,
            'career_level': self._determine_career_level(total_years, job_sections)
        }
    
    def _extract_job_sections(self, text):
        """Extract individual job positions"""
        # This is a simplified version - in reality, you'd use more sophisticated NLP
        lines = text.split('\n')
        positions = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if any(title.lower() in line.lower() for category in self.job_titles_database.values() for title in category):
                position = {
                    'title': line,
                    'company': '',
                    'duration': '',
                    'description': ''
                }
                
                # Try to find company and duration in nearby lines
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not any(title.lower() in next_line.lower() for category in self.job_titles_database.values() for title in category):
                        position['company'] = next_line
                
                positions.append(position)
        
        return positions[:5]  # Return top 5 positions
    
    def _determine_career_level(self, years, positions):
        """Determine career level based on experience"""
        if years >= 10:
            return 'Senior'
        elif years >= 5:
            return 'Mid-level'
        elif years >= 2:
            return 'Junior'
        else:
            return 'Entry-level'
    
    def extract_education(self, text):
        """Extract education information"""
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'degree', 'university',
            'college', 'institute', 'school', 'certification', 'diploma'
        ]
        
        text_lower = text.lower()
        education_info = []
        
        lines = text.split('\n')
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in education_keywords):
                education_info.append(line.strip())
        
        return {
            'degrees': education_info[:3],  # Top 3 education entries
            'has_degree': len(education_info) > 0
        }
    
    def calculate_ats_score(self, text, job_description=''):
        """Calculate ATS (Applicant Tracking System) compatibility score"""
        score = 0
        max_score = 100
        
        # Check for contact information (20 points)
        contact_info = self.extract_contact_info(text)
        if contact_info['emails']:
            score += 10
        if contact_info['phones']:
            score += 10
        
        # Check for skills section (30 points)
        skills = self.extract_skills(text)
        total_skills = sum(len(category_skills) for category_skills in skills.values())
        if total_skills >= 10:
            score += 30
        elif total_skills >= 5:
            score += 20
        elif total_skills > 0:
            score += 10
        
        # Check for experience section (25 points)
        experience = self.extract_experience(text)
        if experience['total_years'] > 0:
            score += 15
        if len(experience['positions']) > 0:
            score += 10
        
        # Check for education (15 points)
        education = self.extract_education(text)
        if education['has_degree']:
            score += 15
        
        # Check for formatting and structure (10 points)
        if len(text.split('\n')) > 10:  # Well-structured
            score += 5
        if len(text.split()) > 100:  # Adequate length
            score += 5
        
        return min(score, max_score)
    
    def generate_recommendations(self, analysis_results):
        """Generate personalized recommendations for CV improvement"""
        recommendations = []
        
        # Contact information recommendations
        if not analysis_results['contact_info']['emails']:
            recommendations.append({
                'category': 'Contact Information',
                'priority': 'High',
                'suggestion': 'Add a professional email address to your CV'
            })
        
        # Skills recommendations
        total_skills = sum(len(skills) for skills in analysis_results['skills'].values())
        if total_skills < 5:
            recommendations.append({
                'category': 'Skills',
                'priority': 'High',
                'suggestion': 'Add more relevant technical and soft skills to strengthen your profile'
            })
        
        # Experience recommendations
        if analysis_results['experience']['total_years'] == 0:
            recommendations.append({
                'category': 'Experience',
                'priority': 'Medium',
                'suggestion': 'Include internships, projects, or volunteer work to demonstrate experience'
            })
        
        # ATS score recommendations
        if analysis_results['ats_score'] < 70:
            recommendations.append({
                'category': 'ATS Optimization',
                'priority': 'High',
                'suggestion': 'Improve CV structure and add more keywords to increase ATS compatibility'
            })
        
        return recommendations

# Initialize the analyzer
cv_analyzer = AdvancedCVAnalyzer()

@cv_bp.route('/analyze_cv', methods=['POST'])
def analyze_cv():
    """Main endpoint for CV analysis"""
    try:
        data = request.get_json()
        cv_text = data.get('text', '')
        job_description = data.get('job_description', '')
        
        if not cv_text:
            return jsonify({
                'status': 'error',
                'message': 'CV text is required'
            }), 400
        
        # Perform comprehensive analysis
        contact_info = cv_analyzer.extract_contact_info(cv_text)
        skills = cv_analyzer.extract_skills(cv_text)
        experience = cv_analyzer.extract_experience(cv_text)
        education = cv_analyzer.extract_education(cv_text)
        ats_score = cv_analyzer.calculate_ats_score(cv_text, job_description)
        
        analysis_results = {
            'contact_info': contact_info,
            'skills': skills,
            'experience': experience,
            'education': education,
            'ats_score': ats_score
        }
        
        recommendations = cv_analyzer.generate_recommendations(analysis_results)
        
        # Calculate overall quality score
        quality_score = min(100, (
            (len(contact_info['emails']) > 0) * 15 +
            (sum(len(skills) for skills in skills.values()) > 5) * 25 +
            (experience['total_years'] > 0) * 20 +
            (education['has_degree']) * 15 +
            (ats_score / 100) * 25
        ))
        
        result = {
            'status': 'success',
            'analysis': analysis_results,
            'recommendations': recommendations,
            'scores': {
                'overall_quality': round(quality_score, 1),
                'ats_compatibility': ats_score,
                'completeness': round((len(recommendations) == 0) * 100, 1)
            },
            'summary': {
                'total_skills': sum(len(skills) for skills in skills.values()),
                'experience_level': experience['career_level'],
                'has_contact_info': len(contact_info['emails']) > 0,
                'has_education': education['has_degree']
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in CV analysis: {error_trace}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'trace': error_trace
        }), 500

@cv_bp.route('/match_job', methods=['POST'])
def match_job():
    """Endpoint for job matching analysis"""
    try:
        data = request.get_json()
        cv_text = data.get('cv_text', '')
        job_description = data.get('job_description', '')
        
        if not cv_text or not job_description:
            return jsonify({
                'status': 'error',
                'message': 'Both CV text and job description are required'
            }), 400
        
        # Analyze both CV and job description
        cv_skills = cv_analyzer.extract_skills(cv_text)
        job_skills = cv_analyzer.extract_skills(job_description)
        
        # Calculate match percentage
        cv_skills_flat = [skill['name'] for category in cv_skills.values() for skill in category]
        job_skills_flat = [skill['name'] for category in job_skills.values() for skill in category]
        
        matching_skills = set(cv_skills_flat) & set(job_skills_flat)
        match_percentage = (len(matching_skills) / max(len(job_skills_flat), 1)) * 100
        
        result = {
            'status': 'success',
            'match_percentage': round(match_percentage, 1),
            'matching_skills': list(matching_skills),
            'missing_skills': list(set(job_skills_flat) - set(cv_skills_flat)),
            'recommendation': 'Strong match' if match_percentage >= 70 else 
                           'Good match' if match_percentage >= 50 else 'Needs improvement'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@cv_bp.route('/health', methods=['GET'])
def cv_health_check():
    """Health check endpoint for CV analysis"""
    return jsonify({
        'status': 'healthy',
        'analyzer_ready': True,
        'version': '2.0.0-advanced'
    })


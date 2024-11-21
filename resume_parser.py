import os
import re
import spacy
import constants as cs
import pandas as pd
from spacy.matcher import Matcher
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords 
from spacy.tokens import Span
from spellchecker import SpellChecker
from collections import Counter
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words
    
# Define section aliases and their identifiers
section_definitions = {
    "summary": {
        "aliases": [
            "personal summary", "summary", "professional summary",
            "experience summary", "professional statement",
            "summary statement", "career overview",
            "experience overview", "profile",
            "profile overview", "profile summary",
            "professional profile", "work summary",
            "work overview"
        ],
        "identifier": ["address", "zip code", "mail id"]
    },
    "statement": {
        "aliases": [
            "statement", "objective", "objective statement",
            "resume objective", "personal objective",
            "profile objective", "personal statement"
        ],
        "identifier": ["address", "zip code", "mail id"]
    },
    "experience": {
        "aliases": [
            "professional experience", "work",
            "work experience", "experience", "training"
        ]
    },
    "education": {
        "aliases": [
            "education", "coursework",
            "schooling"
        ]
    },
    "skills": {
        "aliases": [
            "skills", "skillset"
        ]
    },
    "projects": {
        "aliases": [
            "project", "projects"
        ]
    },
    "achievements": {
        "aliases": [
            "achievements", "accomplishments", "awards"
        ]
    }
}



def extract_entity_sections(nlp_text):
    
    
    sections = {}
    current_section = None
    
    # Function to check for aliases in tokens
    def check_for_aliases(token):
        for section, details in section_definitions.items():
            if token.text.lower() in details["aliases"]:
                return section
        return None

    # Loop through tokens to identify section headers
    for i, token in enumerate(nlp_text):
        if token.text.lower() in [alias for aliases in section_definitions.values() for alias in aliases['aliases']]:
            detected_section = check_for_aliases(token)

            if detected_section:
                # If current section is already set, skip to avoid multiple detections
                if current_section and detected_section in sections:
                    continue
                
                # Collect the section content
                start_index = i
                if detected_section not in sections:
                    sections[detected_section] = ""
                    current_section = detected_section

                # Look for next section or end of document
                for j in range(i + 1, len(nlp_text)):
                    next_section = check_for_aliases(nlp_text[j])
                    if next_section and next_section != detected_section:
                        break
                else:
                    j = len(nlp_text)  # If no new section found, go to the end

                # Capture section text
                sections[detected_section] += nlp_text[start_index:j].text + "\n"
                i = j - 1  # Move index to end of current section

    return sections

def extract_surrounding_text(text):
    # Split the text into lines for easier processing
    text = text.lower()
    text = re.sub(r'\s{6,}', '\n', text)
    lines = text.splitlines()
    
    

    # Define the regex patterns to detect the date ranges
    month_year_pattern = r'(\b(?:\w+\s+\d{4}|\d{4})\b)\s*[-–]\s*(\b(?:\w+\s+\d{4}|present)\b)'
    year_year_pattern = r'(\d{4}) - (\d{4}|present)'

    results = []

    # Iterate through each line to check for the date patterns
    for i in range(len(lines)):
        
        line = lines[i]

        # Check for month-year pattern
        if re.search(month_year_pattern, line):
            # Extract surrounding lines (2 above, 2 below)
            start_index = max(i - 2, 0)  # Ensure we don't go below index 0
            end_index = min(i + 3, len(lines))  # Ensure we don't go beyond the last index
            
            surrounding_lines = lines[start_index:end_index]
            # Filter out lines starting with "•"
            surrounding_lines = [l for l in surrounding_lines if not l.startswith("•")]
            results.append('\n'.join(surrounding_lines) + "\n")

        # Check for year-year pattern
        if re.search(year_year_pattern, line):
            # Extract surrounding lines (2 above, 2 below)
            start_index = max(i - 2, 0)  # Ensure we don't go below index 0
            end_index = min(i + 3, len(lines))  # Ensure we don't go beyond the last index
            
            surrounding_lines = lines[start_index:end_index]
            # Filter out lines starting with "•"
            surrounding_lines = [l for l in surrounding_lines if not l.startswith("•")]
            results.append('\n'.join(surrounding_lines) + "\n")

    return results


def get_experience_section(resume_text):
    sections = extract_entity_sections(resume_text)
    return extract_surrounding_text(sections.get("experience", "No Experience section found."))


def get_education_section(resume_text):
    sections = extract_entity_sections(resume_text)
    return sections.get("education", "No Education section found.")


def extract_email(text):
    """Extracts email from resume text."""
    email = re.findall(r"([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    return email[0].split()[0].strip(';') if email else None

def extract_name(nlp_text, matcher):
    """Extracts the name using Spacy's matcher."""
    pattern = [cs.NAME_PATTERN]
    matcher.add('NAME', None, *pattern)
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        return span.text

def extract_skills(nlp_text):
    """Extracts skills based on a predefined skill list."""
    nlp_text = nlp(nlp_text)
    noun_chunks = list(nlp_text.noun_chunks)
    tokens = [token.text for token in nlp_text if not token.is_stop]
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), 'skills.csv'))
    skills = list(data.columns.values)
    skillset = {token.capitalize() for token in tokens + [chunk.text.lower().strip() for chunk in noun_chunks] if token.lower() in skills}
    return list(skillset)

# def extract_education(nlp_text):
#     """Extracts education details such as degrees and years."""
#     edu = {}
#     for index, text in enumerate(nlp_text):
#         words = text.split()
#         if any(word.upper() in cs.EDUCATION for word in words):
#             edu[words[0]] = text + (nlp_text[index + 1] if index + 1 < len(nlp_text) else '')
#     education = [(key, re.search(cs.YEAR_PATTERN, value).group(0)) for key, value in edu.items() if re.search(cs.YEAR_PATTERN, value)]
#     return education

# def extract_experience(text):
#     """Extracts experience details from the resume text."""
#     lemmatizer = WordNetLemmatizer()
#     tokens = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(text) if word not in stopwords.words('english')]
#     tagged = nltk.pos_tag(tokens)
#     chunk = nltk.RegexpParser('P: {<NNP>+}').parse(tagged)
#     experience = [" ".join(leaf[0] for leaf in subtree.leaves()) for subtree in chunk.subtrees() if subtree.label() == 'P' and 'experience' in subtree[0][0].lower()]
#     return experience

# def extract_competencies(text, experience_list):
#     """Extracts competencies based on keywords."""
#     competencies = {comp: [kw for kw in keywords if string_found(kw, " ".join(experience_list))] for comp, keywords in cs.COMPETENCIES.items()}
#     return {k: v for k, v in competencies.items() if v}

# def extract_measurable_results(text, experience_list):
#     """Extracts measurable results based on keywords."""
#     measurable_results = {mr: [kw for kw in keywords if string_found(kw, " ".join([exp[:len(exp) // 2] for exp in experience_list]))] for mr, keywords in cs.MEASURABLE_RESULTS.items()}
#     return {k: v for k, v in measurable_results.items() if v}

import re
from collections import Counter

from collections import defaultdict

# Define the measurable results dictionary with scoring values

def extract_impact_words(resume_text):
    impact = {
        'metrics': [],
        'action_words': [],
        'weak_words': []
    }
    words = resume_text.lower().split()
    
    # Check for each category of words in the resume text
    for category, keywords in cs.MEASURABLE_RESULTS.items():
        for keyword in keywords:
            keyword_pattern = rf'\b{re.escape(keyword)}\b'
            if re.search(keyword_pattern, resume_text, re.IGNORECASE):
                impact[category].append(keyword)
                
    return impact

# Function to calculate the impact score based on extracted words
def calculate_impact_score(impact_data):
    score = 0
    
    # Tally up the score based on occurrences and weights
    for category, words in impact_data.items():
        score += len(words) * cs.SCORE_VALUES[category]
        
    return score


def positive_buzzwords(resume_content):
    # Normalize the resume content to lowercase for consistent matching
    resume_content = resume_content.lower()

    # Normalize buzzwords to lowercase for case-insensitive matching
    buzzwords_normalized = [word.lower() for word in cs.POSITIVE_BUZZWORDS]

    # Find all words in the resume content
    words = re.findall(r'\b\w+\b', resume_content)

    # Filter words to include only positive buzzwords
    buzzwords = [word for word in words if word in buzzwords_normalized]

    return list(set(buzzwords))

import re

def negative_buzzwords(resume_content):
    # Normalize the resume content to lowercase for consistent matching
    resume_content = resume_content.lower()

    # Normalize buzzwords to lowercase for case-insensitive matching
    buzzwords_normalized = [word.lower() for word in cs.NEGATIVE_BUZZWORDS]

    # Find all words in the resume content
    words = re.findall(r'\b\w+\b', resume_content)

    # Filter words to include only positive buzzwords
    buzzwords = [word for word in words if word in buzzwords_normalized]

    return list(set(buzzwords))


# Function to clean text and remove stop words
def clean_and_tokenize(doc):
    
    tokens = [token.text for token in doc if token.is_alpha and token.text not in stop_words]
    return tokens

# Function to count repetition of tokens in a specific section
def count_word_repetition(doc):
    tokens = [token.text for token in doc if token.is_alpha and token.text not in stop_words]
    tokens = [token.lower() for  token in tokens]
    token_counts = Counter(tokens)
    repeated_words = {word: count for word, count in token_counts.items() if count >= 3}
    return repeated_words

def count_spelling_mistakes(nlp_text):
    spell = SpellChecker()
    
    words = [token.text for token in nlp_text if token.is_alpha]  
    
    misspelled = spell.unknown(words)
    
    # Return the count of misspelled words
    return len(misspelled)

def count_bullet_points(resume_text):
    # Count occurrences of the bullet character "•"
    return resume_text.count("•")

def get_length(nlp_text):
    return len(nlp_text)

def string_found(substring, text):
    """Helper function to check for exact keyword match in text."""
    return bool(re.search(r"\b" + re.escape(substring) + r"\b", text))


def extract_education_details_single_dict(text):
    """
    Extracts structured education information from a given text.

    Parameters:
    - text (str): Text containing education details with institution names, degrees, majors, dates, and GPA.

    Returns:
    - dict: A dictionary with keys (e.g., 'institution', 'time_period', 'degree', 'major', 'gpa') each holding lists of values.
    """
    # Regular expression to capture the relevant information
    pattern = r'(?P<institution>.?)\s+(?P<start_month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\s+(?P<start_year>\d{4})\s-\s*(?P<end_month>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec))\s+(?P<end_year>\d{4})\s+(?P<degree>.?)\s+in\s+(?P<major>.?)\s+(?P<gpa>\d+\.\d+/\d+\.\d+)\s*CGPA|(?P<single_degree>[A-Za-z. ]+)\s+in\s+(?P<single_major>[A-Za-z ]+)'

    # Initialize a dictionary with lists for each key to store multiple entries
    results = {
        "institution": [],
        "time_period": [],
        "degree": [],
        "major": [],
        "gpa": []
    }

    # Use regex to search and extract components
    matches = re.finditer(pattern, text)

    for match in matches:
        if match.group('institution'):  # For structured entries
            institution = match.group('institution').strip()  # Extract institution name
            start_month = match.group('start_month')          # Extract start month
            start_year = match.group('start_year')            # Extract start year
            end_month = match.group('end_month')              # Extract end month
            end_year = match.group('end_year')                # Extract end year
            degree = match.group('degree').strip()            # Extract degree
            major = match.group('major').strip()              # Extract major
            gpa = match.group('gpa')                           # Extract GPA

            # Append each component to its respective list in the results dictionary
            results["institution"].append(institution)
            results["time_period"].append(f"{start_month} {start_year} - {end_month} {end_year}")
            results["degree"].append(degree)
            results["major"].append(major)
            results["gpa"].append(gpa)
        elif match.group('single_degree') and match.group('single_major'):  # For single degree entries
            degree = match.group('single_degree').strip()      # Extract single degree
            major = match.group('single_major').strip()        # Extract major for single degree

            # Append each component to its respective list in the results dictionary
            results["institution"].append("Unknown Institution")  # Default for unknown institution
            results["time_period"].append("Not specified")        # No time period provided
            results["degree"].append(degree)
            results["major"].append(major)
            results["gpa"].append("Not specified")                # No GPA provided

    return results


import re

def extract_experience_details(experience_text):
    """
    Extracts years of experience, designations, and company names from a given experience section text.

    Parameters:
    - experience_text (list): List of strings where each string is a work experience entry.

    Returns:
    - dict: A dictionary with 'total_experience_years', 'designations', and 'companies' each as lists.
    """
    # Initialize lists to store extracted details
    designations = []
    companies = []
    total_experience_years = 0

    # Regular expressions for designations, companies, and experience duration
    designation_pattern = r"(?P<designation>[\w\s]+)\n"           # Extracts designation from start of each entry
    company_pattern = r"\n\n(?P<company>[^,\n]+),"                # Matches company name
    duration_pattern = r"(?P<start_month>\w+)\s(?P<start_year>\d{4})\s–\s(?P<end_month>\w+)\s(?P<end_year>\d{4}|present)"  # Matches start and end dates

    # Month-to-integer mapping for date calculations
    months = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }

    # Iterate through each experience entry
    for entry in experience_text:
        # Extract designation
        designation_match = re.search(designation_pattern, entry, re.IGNORECASE)
        if designation_match:
            designations.append(designation_match.group('designation').strip())

        # Extract company name
        company_match = re.search(company_pattern, entry, re.IGNORECASE)
        if company_match:
            companies.append(company_match.group('company').strip())

        # Calculate experience duration
        duration_match = re.search(duration_pattern, entry, re.IGNORECASE)
        if duration_match:
            start_year = int(duration_match.group('start_year'))
            end_year = duration_match.group('end_year')
            start_month = months[duration_match.group('start_month').lower()]
            end_month = months[duration_match.group('end_month').lower()]

            # Convert "present" to the current year and month
            if end_year.lower() == "present":
                from datetime import datetime
                current_date = datetime.now()
                end_year = current_date.year
                end_month = current_date.month
            else:
                end_year = int(end_year)

            # Calculate duration in years and months
            experience_years = end_year - start_year
            experience_months = end_month - start_month
            if experience_months < 0:
                experience_years -= 1
                experience_months += 12

            # Convert to total years
            total_experience_years += experience_years + (experience_months / 12)

    return {
        "total_experience_years": round(total_experience_years, 1),  # Round to 1 decimal
        "designations": designations,
        "companies": companies
    }

def get_projects(text):
    # Define a list to store matched projects
    projects = []

    # Regular expression pattern:
    # - `\n`: Ensures the sentence starts after a newline.
    # - `(\w+\s?){1,10}`: Matches up to 10 words.
    # - `[: -]?`: Optional colon, space, or dash following the sentence.
    # - `\n•`: Ensures the bullet point appears on the next line.
    pattern = r'\n((?:\w+\s?){1,10})[: -]?\n•'

    # Find all matches in the provided text
    matches = re.findall(pattern, text)

    # Append each matched project to the list
    for match in matches:
        projects.append(match.strip())

    return projects



class ResumeScorer:
    def __init__(self, resume):
        self.resume = resume

    def calculate_score(self):
        score = 100
        
        # Bullet Points
        if 12 <= self.resume['bulletPoints'] <= 20:
            score += 3  # Add full weightage if bullet points are in the optimal range

        # Length
        if 500 <= self.resume['length'] <= 1000:
            score += 3  # Add full weightage for optimal length

        # Repeated Words
        for word, count in self.resume['repeatedWords'].items():
            score -= (count // 5)  # Deduct 1 point for every 5 repetitions

        # Spelling Mistakes
        score -= self.resume['spellingMistakes'] * 4  # Deduct 4 points per mistake

        # Experience
        if self.resume['totalExperience'] > 0:  # Assuming experience is a non-empty list if present
            score += 2

        # Positive Buzzwords
        score += len(self.resume['positiveBuzzwords']) * 1  # 1 point per positive buzzword

        #skills
        skills_score = min(len(set(self.resume['skills'])) // 5, 5)
        score += skills_score
        # Negative Buzzwords
        score -= len(self.resume['negativeBuzzwords']) * 2  # Deduct 2 points per negative buzzword

        # Impact Words (action words and metrics per bullet point)
        total_impact_words = len(self.resume['impactWords']['action_words']) + len(self.resume['impactWords']['metrics'])
        impact_per_bullet_point = total_impact_words / max(self.resume['bulletPoints'], 1)  # Avoid division by zero
        score += impact_per_bullet_point * 4  # Add weightage for impact words per bullet point

        # Certifications
        #score += len(self.resume['certifications']) * 3  # 3 points per certification

        # Projects
        score += len(self.resume['projects']) * 3  # 3 points per project

        # Inclusion of Key Sections
        required_sections = ['projects', 'experience', 'education', 'summary', 'objective', 'achievements']
        for section in required_sections:
            if section in self.resume["sections"]:
                score += 1  # Add 1 point if section is included
            else:
                score -= 1  # Deduct 1 point if section is missing

        # Ensure score is within a reasonable range (e.g., between 0 and 100)
        return score



import re

def calculate_score(details):
    
    # Rule 1: Bullet Points (12-20 is good)
    bullet_points_score = 10 if 12 <= details['bulletPoints'] <= 20 else 0
    bullet_points_weight = 3
    bullet_points_final = bullet_points_score * bullet_points_weight
    
    # Rule 2: Length (500-1000 is good)
    length_score = 10 if 500 <= details['length'] <= 1000 else 0
    length_weight = 3
    length_final = length_score * length_weight
    
    # Rule 3: Repetition Words (-1 point for every 5 repetitions)
    repetition_score = -1 * (sum(details['repeatedWords'].values()) // 5)
    
    # Rule 4: Spelling Mistakes (-4, Weightage: 4)
    spelling_score = -4 * details['spellingMistakes']
    spelling_weight = 4
    spelling_final = spelling_score * spelling_weight
    
    # Rule 5: Experience (+2 or 0 based on if there is experience)
    experience_score = 2 if details['totalExperience'] > 0 else 0
    experience_weight = 2
    experience_final = experience_score * experience_weight
    
    # Rule 6: Positive Buzzwords (+1, Weightage: 3)
    positive_buzzwords_score = len(details['positiveBuzzwords'])
    positive_buzzwords_weight = 3
    positive_buzzwords_final = positive_buzzwords_score * positive_buzzwords_weight
    
    # Rule 7: Negative Buzzwords (-2, Weightage: 4)
    negative_buzzwords_score = -2 * len(details['negativeBuzzwords'])
    negative_buzzwords_weight = 4
    negative_buzzwords_final = negative_buzzwords_score * negative_buzzwords_weight
    
    # Rule 8: Impact Words (Actions words, Metric) per bullet point, Weightage: 4
    impact_words_score = len(details['impactWords']['action_words']) + len(details['impactWords']['metrics'])
    impact_words_weight = 4
    impact_words_final = impact_words_score * impact_words_weight
    
    # Rule 9: Certifications (+3 per certification, Weightage: 4)
    certifications_score = 3 * len(details.get('certifications', []))
    certifications_weight = 4
    certifications_final = certifications_score * certifications_weight
    
    # Rule 10: Projects (+3 per project, Weightage: 4)
    projects_score = 3 * len(details['projects'])
    projects_weight = 4
    projects_final = projects_score * projects_weight
    
    # Rule 11: Inclusion of Sections (Projects/Experience, Education, Summary, Objective, Accomplishments)
    # Check for presence of key sections and give +1 if present, else -1
    sections = [
        details['summary'], 
        details['objective'], 
        details['achievements'], 
        details['degrees'], 
        details['projects']
    ]
    sections_score = sum(1 if section else -1 for section in sections)
    skills_score = min(len(set(details['skills'])) // 5, 5)
    
    # Calculate total score
    total_score = (bullet_points_final + length_final + experience_final + 
                   positive_buzzwords_final + negative_buzzwords_final +
                   impact_words_final + certifications_final + projects_final + 
                   sections_score + repetition_score + spelling_final + skills_score + 100)
    
    # Ensure score is between 1 and 100
    final_score = max(1, min(total_score, 100))
    
    return final_score

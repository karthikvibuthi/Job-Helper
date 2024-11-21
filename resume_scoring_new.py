import os
import re
import spacy
import pandas as pd
from spacy.matcher import Matcher
from collections import Counter
from nltk.corpus import stopwords
from spellchecker import SpellChecker


# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Default stopwords
stop_words = nlp.Defaults.stop_words

# Scoring Weights
SCORE_WEIGHTS = {
    "sections": 20,  # Sections completeness
    "spelling": 10,  # Spell-check
    "keywords": 15,  # Impactful words
    "skills": 20,    # Skills section
    "structure": 25, # Overall structure
    "experience": 15, # Experience section
    "projects": 12,   # Projects section
    "repetition": -5  # Deduct for repeated words
}

# Section Definitions
required_sections = ["Summary", "Experience", "Education", "Skills", "Projects"]

# Extract sections
def extract_sections(text):
    sections = {sec: "" for sec in required_sections}
    current_section = None

    for line in text.splitlines():
        line = line.strip()
        # Match section titles case-insensitively
        for section in required_sections:
            if section.lower() in line.lower():
                current_section = section
                break  # Once the section is matched, stop checking other sections
        if current_section:
            sections[current_section] += line + "\n"

    return sections

# Calculate sections completeness
def sections_completeness(sections):
    filled_sections = sum(1 for sec in required_sections if sections[sec].strip())
    return filled_sections / len(required_sections) * SCORE_WEIGHTS["sections"]

# Extract skills
def extract_skills(nlp_text):
    tokens = [token.text for token in nlp_text if not token.is_stop]
    skills = pd.read_csv('skills.csv').columns.tolist()
    print(f"Tokens: {tokens}")
    detected_skills = set(token.lower() for token in tokens if token.lower() in skills)
    print(f"Detected Skills: {detected_skills}")
    return detected_skills

# Skills scoring
def skills_score(skills_detected):
    if len(skills_detected) > 1:
        return SCORE_WEIGHTS["skills"]
    return (len(skills_detected) / 5) * SCORE_WEIGHTS["skills"]

# Extract spelling mistakes
def count_spelling_mistakes(nlp_text):
    spell = SpellChecker()
    words = [token.text for token in nlp_text if token.is_alpha]
    misspelled = spell.unknown(words)
    print(f"Words: {words}")
    print(f"Misspelled Words: {misspelled}")
    return len(misspelled)

# Spelling score
def spelling_score(misspelled_count):
    if misspelled_count == 0:
        return SCORE_WEIGHTS["spelling"]
    elif misspelled_count < 5:
        return (1 - misspelled_count / 5) * SCORE_WEIGHTS["spelling"]
    return 0

# Count impactful words (optional: use your own set of impactful keywords)
def count_impactful_words(text):
    impact_words = [
        'developed',
        'led',
        'analyzed',
        'collaborated',
        'conducted',
        'performed',
        'recruited',
        'improved',
        'founded',
        'transformed',
        'composed',
        'conceived',
        'designed',
        'devised',
        'established',
        'generated',
        'implemented',
        'initiated',
        'instituted',
        'introduced',
        'launched',
        'opened',
        'originated',
        'pioneered',
        'planned',
        'prepared',
        'produced',
        'promoted',
        'started',
        'released',
        'administered',
        'assigned',
        'chaired',
        'consolidated',
        'contracted',
        'co-ordinated',
        'delegated',
        'directed',
        'evaluated',
        'executed',
        'organized',
        'oversaw',
        'prioritized',
        'recommended',
        'reorganized',
        'reviewed',
        'scheduled',
        'supervised',
        'guided',
        'advised',
        'coached',
        'demonstrated',
        'illustrated',
        'presented',
        'taught',
        'trained',
        'mentored',
        'spearheaded',
        'authored',
        'accelerated',
        'achieved',
        'allocated',
        'completed',
        'awarded',
        'persuaded',
        'revamped',
        'influenced',
        'assessed',
        'clarified',
        'counseled',
        'diagnosed',
        'educated',
        'facilitated',
        'familiarized',
        'motivated',
        'participated',
        'provided',
        'referred',
        'rehabilitated',
        'reinforced',
        'represented',
        'moderated',
        'verified',
        'adapted',
        'coordinated',
        'enabled',
        'encouraged',
        'explained',
        'informed',
        'instructed',
        'lectured',
        'stimulated',
        'classified',
        'collated',
        'defined',
        'forecasted',
        'identified',
        'interviewed',
        'investigated',
        'researched',
        'tested',
        'traced',
        'interpreted',
        'uncovered',
        'collected',
        'critiqued',
        'examined',
        'extracted',
        'inspected',
        'inspired',
        'summarized',
        'surveyed',
        'systemized',
        'arranged',
        'budgeted',
        'controlled',
        'eliminated',
        'itemised',
        'modernised',
        'operated',
        'organised',
        'processed',
        'redesigned',
        'reduced',
        'refined',
        'resolved',
        'revised',
        'simplified',
        'solved',
        'streamlined',
        'appraised',
        'audited',
        'balanced',
        'calculated',
        'computed',
        'projected',
        'restructured',
        'modelled',
        'customized',
        'fashioned',
        'integrated',
        'proved',
        'revitalized',
        'set up',
        'shaped',
        'structured',
        'tabulated',
        'validated',
        'approved',
        'catalogued',
        'compiled',
        'dispatched',
        'filed',
        'monitored',
        'ordered',
        'purchased',
        'recorded',
        'retrieved',
        'screened',
        'specified',
        'systematized',
        'conceptualized',
        'brainstomed',
        'tasked',
        'supported',
        'proposed',
        'boosted',
        'earned',
        'negotiated',
        'navigated',
        'updated',
        'utilized'
    ]
    return sum(1 for word in impact_words if word in text.lower())

# Impactful words score
def impactful_words_score(count):
    max_score = 10
    return min(count / 10, 1) * SCORE_WEIGHTS["keywords"]

# Structure evaluation (bonus if formatted well)
def structure_score(text):
    bullet_points = text.count("•")
    if bullet_points > 10:
        return SCORE_WEIGHTS["structure"]
    return (bullet_points / 10) * SCORE_WEIGHTS["structure"]

# Check for repetitive words and penalize
def check_repeated_words(text):
    words = [word.lower() for word in text.split() if word.isalpha()]
    word_count = Counter(words)
    repeated_words = {word: count for word, count in word_count.items() if count > 3}  # Threshold is 3 repetitions
    return repeated_words

# Repetition word score
def repetition_word_score(repeated_words):
    if repeated_words:
        return SCORE_WEIGHTS["repetition"]
    return 0

# Main function to calculate resume score
def calculate_resume_score(resume_text):
    """
    Calculates the resume score based on various factors.

    Parameters:
    pdf_path (str): Path to the resume PDF file.

    Returns:
    dict: A dictionary containing the total score and a detailed breakdown of individual scores.
    """
    nlp_text = nlp(resume_text)

    # Extract sections
    sections = extract_sections(resume_text)

    # Section completeness
    section_score = sections_completeness(sections)

    # Skills detection
    skills_detected = extract_skills(nlp_text)
    skills_eval_score = skills_score(skills_detected)

    # Spelling check
    misspelled_count = count_spelling_mistakes(nlp_text)
    spelling_eval_score = spelling_score(misspelled_count)

    # Count impactful words
    impactful_words_count = count_impactful_words(resume_text)
    impactful_words_eval_score = impactful_words_score(impactful_words_count)

    # Structure scoring
    structure_eval_score = structure_score(resume_text)

    # Check for repeated words
    repeated_words = check_repeated_words(resume_text)
    repetition_eval_score = repetition_word_score(repeated_words)

    # Calculate total score
    total_score = section_score + skills_eval_score + spelling_eval_score + impactful_words_eval_score + structure_eval_score + repetition_eval_score

    # Create the result object
    result = {
        "total_score": round(total_score, 2),
        "breakdown": {
            "sections_completeness": round(section_score, 2),
            "skills_evaluation": round(skills_eval_score, 2),
            "spelling_evaluation": round(spelling_eval_score, 2),
            "impactful_words": round(impactful_words_eval_score, 2),
            "structure_evaluation": round(structure_eval_score, 2),
            "repetition_evaluation": round(repetition_eval_score,2),
        },
    }

    return result

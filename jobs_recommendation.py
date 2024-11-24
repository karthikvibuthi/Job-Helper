import pandas as pd

def get_job_recommendations(resume_skills_list, job_list_csv):
    """
    Generates job recommendations based on skill match scores.

    Parameters:
    resume_skills_list (list of str): A list of skills from the resume.
    job_list_csv (str): Path to the job listings CSV file.

    Returns:
    list of dict: A list of job recommendations sorted by match score, each containing:
                  id, posting_url, job_title, and employer_name.
    """

    # Load job listings
    job_listings_df = pd.read_csv(job_list_csv)

    # Combine the resume skills into a single string for matching
    resume_skills = ', '.join(resume_skills_list).lower()
    resume_skills_set = set(resume_skills.split(','))

    # Define a function to calculate the skill matching score for each job
    def calculate_skill_match(job_skills):
        # Check if job_skills is a string and handle missing values
        if isinstance(job_skills, str):
            job_skills_set = set(job_skills.lower().split(","))
        else:
            job_skills_set = set()  # No skills if job_skills is not a string
        
        # Calculate the number of matching skills
        matching_skills = job_skills_set.intersection(resume_skills_set)
        
        # Calculate match score as the ratio of matching skills to total job skills
        match_score = len(matching_skills) / len(job_skills_set) if job_skills_set else 0
        return match_score, matching_skills

    # Prepare job recommendations
    job_recommendations = []

    # Iterate over each job listing to calculate match scores
    for _, job_row in job_listings_df.iterrows():
        job_id = job_row['id']
        job_title = job_row['job_title']
        employer_name = job_row['employer_name']
        posting_url = job_row['posting_url']
        job_skills = job_row['technical_skills']

        # Calculate match score and matching skills for the job
        match_score, matching_skills = calculate_skill_match(job_skills)
        
        # Store the recommendation details if match_score is positive
        if match_score > 0.3:
            job_recommendations.append({
                'id': job_id,
                'posting_url': posting_url,
                'job_title': job_title,
                'employer_name': employer_name,
                'match_score': match_score,
                'matching_skills': ', '.join(matching_skills)
            })

    # Sort recommendations by match score in descending order and return the top results
    sorted_recommendations = sorted(job_recommendations, key=lambda x: x['match_score'], reverse=True)
    
    return sorted_recommendations

from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Initialize the SentenceTransformer model globally
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded successfully.")

# Global variables to store precomputed embeddings and job listings
encoded_job_embeddings = None
job_listings_df = None  # To store the job DataFrame for reuse

def initialize_job_embeddings(job_list_csv1,job_list_csv2):
    """
    Preloads and encodes the job descriptions from a CSV file.

    Parameters:
    job_list_csv (str): Path to the job listings CSV file.

    Returns:
    None: The function updates the global `encoded_job_embeddings` and `job_listings_df`.
    """
    global encoded_job_embeddings, job_listings_df

    # Load job listings
    try:
        job_listings_df1 = pd.read_csv(job_list_csv1)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {job_list_csv1}")
    except Exception as e:
        raise Exception(f"Error loading the CSV file: {str(e)}")
    
    try:
        job_listings_df2 = pd.read_csv(job_list_csv2)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find the file: {job_list_csv2}")
    except Exception as e:
        raise Exception(f"Error loading the CSV file: {str(e)}")


    # Define required columns
    required_columns = {'id', 'job_title', 'employer_name', 'posting_url', 'requirements', 'technical_skills', 'company_type'}

    # Check if required columns exist in both dataframes
    if not required_columns.issubset(job_listings_df1.columns):
        raise ValueError(f"The job_listings1 CSV file must contain the following columns: {required_columns}")
    if not required_columns.issubset(job_listings_df2.columns):
        raise ValueError(f"The job_listings2 CSV file must contain the following columns: {required_columns}")

    # Retain only the required columns in both dataframes
    job_listings_df1 = job_listings_df1[list(required_columns)]
    job_listings_df2 = job_listings_df2[list(required_columns)]

    # Merge the dataframes
    job_listings_df = pd.concat([job_listings_df1, job_listings_df2], ignore_index=True)


    # Encode job descriptions (requirements column)``
    print("Encoding job descriptions...")
    job_descriptions = job_listings_df['requirements'].fillna('').tolist()
    encoded_job_embeddings = model.encode(job_descriptions, convert_to_tensor=True)
    print("Job descriptions encoding complete and stored globally.")


def get_job_recommendations_sentence_transformer(resume_text):
    """
    Generates job recommendations based on cosine similarity of Sentence-BERT embeddings.

    Parameters:
    resume_text (list of str): A list of skills or relevant text from the resume.

    Returns:
    list of dict: A list of job recommendations sorted by match score, each containing:
                  id, posting_url, job_title, employer_name, match_score, and matching_skills.
    """
    global encoded_job_embeddings, job_listings_df

    # Check if job embeddings have been initialized
    if encoded_job_embeddings is None or job_listings_df is None:
        raise ValueError("Job embeddings are not initialized. Call `initialize_job_embeddings` first.")

    # Encode the resume text
    print("Encoding resume text...")
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    print("Resume encoding complete.")

    # Calculate cosine similarity between the resume and each job description
    print("Calculating cosine similarity...")
    cosine_scores = util.cos_sim(resume_embedding, encoded_job_embeddings)

    # Prepare job recommendations
    job_recommendations = []
    for idx, job_row in job_listings_df.iterrows():
        match_score = cosine_scores[0][idx].item()  # Extract the cosine similarity score

        if match_score > 0.3:  # Threshold for considering a good match
            job_recommendations.append({
                'id': job_row['id'],
                'posting_url': job_row['posting_url'],
                'job_title': job_row['job_title'],
                'employer_name': job_row['employer_name'],
                'match_score': match_score,
                'matching_skills': job_row['technical_skills'] if not pd.isna(job_row['technical_skills']) else '',
                'company_type': job_row['company_type']
                
            })

    # Sort recommendations by match score in descending order and return the results
    sorted_recommendations = sorted(job_recommendations, key=lambda x: x['match_score'], reverse=True)

    print("Job recommendations generated successfully.")
    return sorted_recommendations

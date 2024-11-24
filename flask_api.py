from flask import Flask, request, jsonify, render_template
import app as parsing_module
from events_recommendations import initialize_event_embeddings
from jobs_recommendation import initialize_job_embeddings

events_list_csv = "events_list_latest_3.csv"
initialize_event_embeddings(events_list_csv)

job_list_csv = 'job_listings_latest_skills.csv'
initialize_job_embeddings(job_list_csv)
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # This will render the index.html file from the templates folder

@app.route('/api/parse-resume', methods=['POST'])
def parse_resume():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Call your resume details parsing function
    resume_data = parsing_module.extract_resume_details(file)  # Adjust according to your function
    return jsonify(resume_data)

@app.route('/api/match-jobs-events', methods=['POST'])
def match_jobs_and_events():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Call your matching function
    match_data = parsing_module.extract_match_jobs_and_events(file)  # Adjust according to your function
    return jsonify(match_data)

if __name__ == '__main__':
    app.run(debug=True)

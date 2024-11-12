from flask import Flask, request, jsonify, render_template
import app as parsing_module

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')  # Or specify your domain
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')  # Include any other headers you may need
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response

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

    # Here, you would call your resume parsing function
    resume_data = parsing_module.extract_resume_details(file)  # Adjust according to your function

    return jsonify(resume_data)

@app.route("/api/test")
def helloWorld():
    return "Hello, cross-origin-world!"

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, render_template, request
import pickle
import pandas as pd

with open('model.pkl', 'rb') as f:
    model, le_edu, le_role, le_loc = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = int(request.form['experience'])
    education = le_edu.transform([request.form['education']])[0]
    job_role = le_role.transform([request.form['job_role']])[0]
    location = le_loc.transform([request.form['location']])[0]

    input_df = pd.DataFrame([[experience, education, job_role, location]],
                            columns=['Experience', 'Education', 'Job Role', 'Location'])

    prediction = model.predict(input_df)[0]
    return render_template('index.html', prediction_text=f'Estimated Salary: ₹{round(prediction, 2)} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)

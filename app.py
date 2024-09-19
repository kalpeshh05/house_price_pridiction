from flask import Flask,render_template,request
import pandas as pd
import joblib
import numpy as np


app = Flask(__name__)
data = pd.read_csv("cleaned_data.csv")
pipe = joblib.load("./models/linear_model.joblib")


@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', location=locations)



@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get('location')
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = float(request.form.get('total_sqft'))

        input_data = pd.DataFrame([[location, sqft, bath, bhk]], 
                                  columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_data)[0]
        return str(np.round(prediction, 2))
    except Exception as e:
        return f"Error: {str(e)}. Please check your input values."



if __name__ == '__main__':
    app.run(debug=True)
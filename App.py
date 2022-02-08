from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)


def init():
    # load the saved model.
    global  predictionmodal
    predictionmodal = joblib.load("LogisticRegression.pkl")


@app.route('/')
def welcome():
    return render_template('index.html')



@app.route('/submit', methods=['POST', 'GET'])
def submit():
    total_score = 0
    try:
        if request.method == 'POST':
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            currentSmoker = int(request.form['currentSmoker'])
            cigsPerDay = float(request.form['cigsPerDay'])
            BPMedication = float(request.form['BPMedication'])
            prevalentStroke	= int(request.form['prevalentStroke'])
            hypertensionprevalence = int(request.form['hypertensionprevalence'])
            diabetes = int(request.form['diabetes'])
            totChol = float(request.form['totChol'])
            sysBP = float(request.form['sysBP'])
            diaBP = float(request.form['diaBP'])
            BodyMaskIndex = float(request.form['bodyMaskIndex'])
            heartRate = float(request.form['heartRate'])
            glucose = float(request.form['glucose'])
            
            # Predict Apparent temperature
             # Same order as the x_train dataframe
            features = [np.array([age, sex, currentSmoker, cigsPerDay, BPMedication, prevalentStroke, hypertensionprevalence, diabetes, totChol, sysBP, diaBP, BodyMaskIndex, heartRate, glucose ])]
            prediction = predictionmodal.predict(features)
            finalresult = ''
            if prediction == 0:
               finalresult ='Congratulations!! You Are Healthy'
            else:
                finalresult ='Opps!! Sorry To Say Contact Your Doctor'
            return render_template('index.html', result = finalresult)
    except Exception as e:
        print(e)
        return 'Calculation Error' + str(e), 500

if __name__ == '__main__':
    init()
    app.run(debug=True)

from flask import Flask, render_template, url_for, request
import numpy as np
import pickle

# Load model
model = 'Diabetes_Risk_Model.pkl'
classifier = pickle.load(open(model, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')
    
@app.route('/d_predict')
def d_p():
	return render_template('d_predict.html')

@app.route('/d_verdict', methods=['POST'])
def d_v():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        age = int(request.form['age'])
        
        entries = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        verdict = clf.predict(entries)
        
        return render_template('d_verdict.html', prediction=verdict)
    
@app.route('/readmission')
def r_p():
	return render_template('readmission.html')
    
@app.route('/r_verdict')
def r_v():
	return render_template('r_verdict.html')

if __name__ == '__main__':
	app.run(debug=True)
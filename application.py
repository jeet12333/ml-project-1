import pickle
import os
from flask import Flask, request, render_template, jsonify

application = Flask(__name__)
app = application

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models')

ridge_model = pickle.load(open(os.path.join(models_dir, 'ridge.pkl'), 'rb'))
standard_scaler = pickle.load(open(os.path.join(models_dir, 'scaler.pkl'), 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))
        
        new_data_scaled = standard_scaler.transform(
            [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
        )

        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html', results=result[0])

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0")

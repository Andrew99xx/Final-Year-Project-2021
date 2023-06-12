from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('prediction.html')


@app.route('/predict', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        house=int(request.form['h_area'])
        bed=int(request.form['h_room'])
        storey=int(request.form['h_storey'])
        park=int(request.form['h_parking'])
        gard=int(request.form['h_garea'])
        int_features = [house, bed, storey, park, gard]
    final = [np.array(int_features, dtype=float)]
    prediction = model.predict(final)
    final_output = round(prediction[0], 2)

    return "Your house price is %f" %prediction


if __name__ == "__main__":
    app.run(debug=True)

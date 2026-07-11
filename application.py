from flask import Flask, render_template, request
import pickle
import numpy as np

application = Flask(__name__)
app = application

ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
standard_scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():

    if request.method == "GET":
        return render_template("home.html")

    else:
        Temperature = float(request.form["Temperature"])
        RH = float(request.form["RH"])
        Ws = float(request.form["Ws"])
        Rain = float(request.form["Rain"])
        FFMC = float(request.form["FFMC"])
        DMC = float(request.form["DMC"])
        DC = float(request.form["DC"])
        ISI = float(request.form["ISI"])
        Classes = float(request.form["Classes"])
        Region = float(request.form["Region"])

        data = np.array([[Temperature, RH, Ws, Rain,
                          FFMC, DMC, DC, ISI,
                          Classes, Region]])

        scaled_data = standard_scaler.transform(data)

        prediction = ridge_model.predict(scaled_data)

        return render_template(
            "home.html",
            results=round(prediction[0], 2)
        )


if __name__ == "__main__":
    app.run(debug=True)
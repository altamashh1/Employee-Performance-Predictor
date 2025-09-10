import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load trained model pipeline
with open("models/gwp.pkl", "rb") as f:
    model_data = pickle.load(f)
    pipeline = model_data["pipeline"]
    meta = model_data["meta"]


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict")
def predict():
    return render_template("predict.html")


@app.route("/submit", methods=["POST"])
def submit():
    if request.method == "POST":
        import pandas as pd

        # Collect form data into a dictionary
        input_data = {}
        for feature in meta["feature_names"]:
            val = request.form.get(feature, 0)
            try:
                val = float(val)
            except ValueError:
                pass
            input_data[feature] = [val]   # make it a list so DataFrame gets a column

        # Create a DataFrame with the same columns as training
        df_input = pd.DataFrame(input_data)

        # Make prediction
        prediction = pipeline.predict(df_input)[0]

        return render_template(
            "submit.html",
            prediction=round(float(prediction), 4)
        )
if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask
import pickle
from flask import request, jsonify

# Load the model
model_file = "model2.bin"
dict_vectorizer_file = "dv.bin"

# Load the model
with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

# Load the dict_vectorizer
with open(dict_vectorizer_file, "rb") as f_in:
    dv = pickle.load(f_in)


app = Flask("subscription")


@app.route("/predict", methods=["POST"])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]

    result = {"subscription_probability": float(y_pred)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)

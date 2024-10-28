import pickle

# Load the model
model_file = "model1.bin"
dict_vectorizer_file = "dv.bin"

with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

with open(dict_vectorizer_file, "rb") as f_in:
    dv = pickle.load(f_in)

customer = {"job": "management", "duration": 400, "poutcome": "success"}

X = dv.transform([customer])
y_pred = model.predict_proba(X)[0, 1]

print(f"Score {y_pred}")

# Deployment - ML

## Question 1 and 2

we are using python 3.11 in this homework.
If there is another python installed, use https://www.python.org/downloads/release/python-31110/
Install it using these commands in mac:

- Extract the tar

```bash
tar -xvzf Python-3.11.x.tgz
cd Python-3.11.x
./configure --enable-optimizations --prefix="your_folder_name_to_install_python"
make -j4
sudo make install
```

Install pipenv from using the

```bash
your_folder_name_to_install_python/local/bin/pip3 install pipenv
pipenv --version
pipenv shell
pipenv install scikit-learn==1.5.2 flask gunicorn requests
```

- The **pipenv version** is **2024.2.0**
- The **SHA of scikit learn** is **"sha256:03b6158efa3faaf1feea3faa884c840ebd61b6484167c711548fce208ea09445"**

## Models

There are already prepared dictionary vectorizer and a model. They were trained (roughly) using this code:

```python
features = ['job', 'duration', 'poutcome']
dicts = df[features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X = dv.fit_transform(dicts)

model = LogisticRegression().fit(X, y)
```

We download the model and dict_vectorizer (for one-hot encoding):
With wget:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/master/cohorts/2024/05-deployment/homework
wget $PREFIX/model1.bin
wget $PREFIX/dv.bin
```

Let's use these models!

## Question 3

- First, we write a script for loading these models with pickle and predict the probability of a given client getting a subscriotion or not.

```python
import pickle

# Load the model
model_file = "model1.bin"
dict_vectorizer_file = "dv.bin"
customer = {"job": "management", "duration": 400, "poutcome": "success"}

# Load the model
with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)

# Load the dict_vectorizer
with open(dict_vectorizer_file, "rb") as f_in:
    dv = pickle.load(f_in)

# Transform the json vectorizer to one-hot encoding format
X = dv.transform([customer])

# predict the probability of the model
y_pred = model.predict_proba(X)[0, 1]

print(f"Score {y_pred.round(3)}")
```

We get **Score: 0.759**

## Question4:

Let's serve the above model as a web service. We make two files

1. Q4-server_predict.py
2. Q4-client_predict-test.py

#### Q4-server_predict.py

```python
from flask import Flask
import pickle
from flask import request, jsonify

# Load the model
model_file = "model1.bin"
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
```

#### Q4-client_predict-test.py

````python

```import requests

url = "http://localhost:9696/predict"

client = {"job": "student", "duration": 280, "poutcome": "failure"}
response = requests.post(url, json=client).json()

print(f"client: {client}, response: {response}")
````

We get **subscription_probability: 0.335**

## Docker

We use given image for our next predictions
`svizor/zoomcamp-model:3.11.5-slim`.
This image has this basic structure

```Docker
FROM python:3.11.5-slim
WORKDIR /app
COPY ["model2.bin", "dv.bin", "./"]
```

We pull the image

```bash
docker pull svizor/zoomcamp-model:3.11.5-slim
```

## Question 5

With `docker images`, we get the **size of docker image is 130MB**.

## Question 6

Now we create our own Dockerfile based on the image we pulled.

We made a Docker file as below:

```Docker
FROM svizor/zoomcamp-model:3.11.5-slim

# WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv

RUN pipenv install --system --deploy

COPY ["Q4-server_predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "Q4-server_predict:app"]
```

Since the given image `svizor/zoomcamp-model:3.11.5-slim ` has model named as model2.bin,
we change model name in `Q4-server_predict.py` file and the updated file is as below

#### Q4-server_predict.py

```python
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

```

We change `Q4-client_predict-test.py` for the new client as below

```python
import requests

url = "http://localhost:9696/predict"

client = {"job": "management", "duration": 400, "poutcome": "success"}
response = requests.post(url, json=client).json()

print(f"client: {client}, response: {response}")
```

We build the Docker image

```Docker
docker build -t subscription-prediction .
```

And we run the Docker image

```Docker
docker run -it --rm -p 9696:9696 subscription-prediction
```

After running the `Q4-client_predict-test.py` file, we get

**Subscription_probability: 0.757**

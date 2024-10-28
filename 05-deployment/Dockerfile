FROM svizor/zoomcamp-model:3.11.5-slim

# WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv

RUN pipenv install --system --deploy

COPY ["Q4-server_predict.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "Q4-server_predict:app"]
FROM python:3.9-slim-buster
WORKDIR /app

COPY . .
# COPY ./dataset /app
# COPY ./flask_api /app
# COPY ./models /app
# COPY ./requirements.txt /app

RUN pip install --no-cache -r requirements.txt

EXPOSE 5000
CMD ["python3", "flask_api/app.py"]
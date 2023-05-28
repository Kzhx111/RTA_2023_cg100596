
FROM python:3.10-slim-bullseye

WORKDIR /app

COPY app.py .
COPY model.pkl .
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]

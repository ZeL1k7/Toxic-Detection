FROM python:3.7

ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /app
COPY . .

ENV PORT 8080

CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 8 --timeout 0 main:app
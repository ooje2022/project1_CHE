FROM  python:3.8
WORKDIR /app
COPY requirements.text .
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "/app.py"] 

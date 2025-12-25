FROM python:3.10-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "main.py"]

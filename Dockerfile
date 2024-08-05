FROM python:3.11-bullseye
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc
ADD main.py .
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "./main.py"]
FROM python:3.13.0b4-slim-bullseye
# Or any preferred Python version.
ADD main.py .
COPY . .
RUN pip install -r requirements.txt
CMD ["python", "./main.py"]
# Or enter the name of your unique directory and parameter set.
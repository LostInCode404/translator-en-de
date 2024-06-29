# Start from slim debian python image
FROM python:3.11.9-slim-bookworm

# Install python dependencies
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Setup app dir
RUN mkdir /app
WORKDIR /app
COPY ./*.py ./

# Run app
ENTRYPOINT [ "streamlit", "run", "main.py" ]
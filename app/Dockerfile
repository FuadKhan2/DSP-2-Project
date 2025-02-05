# Use Python 3.13.1 slim image
FROM python:3.13.1-slim

# Set the working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

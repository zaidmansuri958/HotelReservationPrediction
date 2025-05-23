# Use a base Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app


RUN apt-get update && apt-get install -y libgomp1

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy only the necessary files
COPY . .

# Expose the Flask port
EXPOSE 8080

# Run your Flask app
CMD ["python", "application.py"]

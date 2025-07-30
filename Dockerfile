# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install only the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variable for Heroku (port 8080)
ENV PORT=8080
EXPOSE 8080

# Run the Flask app with Gunicorn
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:8080"]


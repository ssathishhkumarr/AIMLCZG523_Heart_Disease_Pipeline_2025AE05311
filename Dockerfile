# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
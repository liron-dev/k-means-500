FROM python:3.10-slim

# Install necessary build tools for compiling specific Python packages
RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Beanstalk maps port 80 to the exposed port by default
EXPOSE 5000

# Running the app using the built-in server to ensure run_pipeline() triggers
CMD ["python", "app.py"]

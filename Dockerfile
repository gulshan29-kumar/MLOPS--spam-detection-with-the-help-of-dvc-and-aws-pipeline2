# Use official python slim image as basis
FROM python:3.10-slim

# Prevent python from writing pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies if required
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Proactively download NLTK data for preprocessing
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copy the project files
COPY . .

# Set default execution command to run DVC pipeline
CMD ["dvc", "repro"]

# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install critical system dependencies
# - libgomp1: GNU OpenMP library, for runtime performance of numerical libraries
# build-essential removed to make the image leaner, as --only-binary=:all: is used for pip

COPY requirements.txt ./

# Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Define a directory where packages will be installed
ENV PKG_DIR=/app/packages
RUN mkdir -p ${PKG_DIR}

# Install Python dependencies to the target directory
# --platform and --only-binary=:all: ensure Linux-compatible wheels are used
RUN pip install \
    --no-cache-dir \
    -r requirements.txt \
    --platform manylinux2014_x86_64 \
    --only-binary=:all: \
    --upgrade \
    --target=${PKG_DIR}

# Add the custom package directory to PYTHONPATH
ENV PYTHONPATH=${PKG_DIR}:${PYTHONPATH:-}
ENV PATH=${PKG_DIR}/bin:${PATH:-}

# Copy the application code and data into the container
COPY backend/ /app/backend/
COPY data/ /app/data/
COPY models/ /app/models/
COPY frontend/ /app/frontend/

# Make port 5000 available
EXPOSE 5000

# Define environment variables
ENV PYTHONUNBUFFERED=1
# JWT_SECRET_KEY should be set as an environment variable in Cloud Run

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "backend.app:app"]

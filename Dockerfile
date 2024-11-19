# Use an official Python 3.12.7 slim image as the base
FROM python:3.12.7-slim

# Install necessary system dependencies for PyTorch, scientific libraries, and Graphviz
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libnss3-dev \
    libgdbm-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    graphviz \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Copy the requirements file into the container
COPY requirements.txt /tmp/requirements.txt

# Install required Python packages from requirements file
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Set the working directory
WORKDIR /workspace

# Set the default command
CMD ["bash"]


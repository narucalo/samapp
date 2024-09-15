# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Prevent interaction prompts during package installation
ENV NAME="samapp"


# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any necessary dependencies
# Install OpenGL dependencies
RUN apt-get update && \
    apt-get install -y tzdata libgl1-mesa-glx libglib2.0-0 && \
    ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# Expose port 8080 (adjust if necessary)
EXPOSE 8080

# Define environment variable
ENV NAME samapp

# Run the application
CMD ["python", "main.py"]

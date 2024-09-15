#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "Docker is not installed. Installing Docker Desktop..."

    # Download Docker Desktop for MacOS
    curl -o ~/Downloads/Docker.dmg https://desktop.docker.com/mac/stable/Docker.dmg

    # Mount the dmg file
    hdiutil attach ~/Downloads/Docker.dmg

    # Copy Docker.app to the Applications folder
    sudo cp -R /Volumes/Docker/Docker.app /Applications

    # Eject the dmg file
    hdiutil detach /Volumes/Docker

    # Open Docker.app to start installation
    open /Applications/Docker.app

    echo "Docker is installed. Please complete the setup manually if prompted, then run this script again."
    exit 1
fi

# Wait for Docker to start if it's not already running
echo "Checking Docker status..."
while (! docker stats --no-stream ); do
    echo "Waiting for Docker to launch..."
    sleep 5
done

# Pull the latest samapp Docker image from Docker Hub
echo "Pulling the samapp Docker image..."
docker pull narucalo2024/samapp:latest

# Run the samapp container
echo "Running samapp..."
docker run -it --rm -p 8080:8080 narucalo2024/samapp:latest

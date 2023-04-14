# Use a lightweight base image with minimal packages
FROM debian:bullseye-slim

# Install necessary packages for OpenCV and other dependencies
RUN apt-get update && \
    apt-get install -y python3-opencv libopencv-dev python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up a non-root user for running the application
RUN useradd -ms /bin/bash appuser
USER appuser

# Copy the application code into the container
COPY app /home/appuser/app

# Install Python dependencies
RUN pip3 install -r /home/appuser/app/requirements.txt

# Set the working directory to the application directory
WORKDIR /home/appuser/app

# Expose the necessary ports
EXPOSE 8000

# Start the application
CMD ["python3", "app.py"]
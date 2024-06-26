# Use the official Python image from the Docker Hub
FROM python:3

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container at /app
ADD indexing.py /app/indexing.py
ADD retrieve.py /app/retrieve.py
ADD re-rank.py /app/re-rank.py
ADD requirements.txt /app/requirements.txt
ADD run.sh /app/run.sh

# Install any needed packages specified in requirements.txt
RUN pip3 install -r requirements.txt

# Make the run.sh script executable
RUN chmod +x /app/run.sh

# Execute the run.sh script
ENTRYPOINT [ "/app/run.sh" ]


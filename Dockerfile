
FROM ubuntu

# ...

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \

    apt-get -y install gcc mono-mcs && \

    rm -rf /var/lib/apt/lists/*

# Use an official Python runtime as a parent image

FROM python:3.6.8



# Set the working directory to /app

WORKDIR /app



# Copy the current directory contents into the container at /app

COPY . /app



# Install any needed packages specified in requirements.txt

RUN pip install flask
RUN pip install pandas
RUN pip install skearn
RUN pip install re
RUN pip install json
RUN pip install nltk



EXPOSE 5000

CMD ["python", "app.py"]
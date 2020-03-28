FROM jjanzic/docker-python3-opencv
RUN pip install --upgrade pip

RUN mkdir app
COPY ./requirements.txt ./app/
RUN pip install -r /app/requirements.txt

COPY ./*.py ./app/

# Set the working directory
WORKDIR /app

CMD python main.py

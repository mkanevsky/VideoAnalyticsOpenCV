FROM jjanzic/docker-python3-opencv
RUN pip install --upgrade pip
RUN pip install requests

RUN mkdir app
COPY ./*.py ./app/

# Set the working directory
WORKDIR /app

ENV VIDEO_PATH rtsp://username:password@ipaddress:554//h264Preview_01_sub
ENV VERBOSE=True
ENV IMAGE_PROCESSING_ENDPOINT=https://your_form_recognizer_service_name.cognitiveservices.azure.com/

CMD python main.py

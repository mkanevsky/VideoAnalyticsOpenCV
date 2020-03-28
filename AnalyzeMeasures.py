from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import TextOperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import TextRecognitionMode
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials

from array import array
import os
from PIL import Image
import sys
import time
import io
import json
import requests
import re
import cv2
import numpy as np


#TODO: incorporate along the way:
# os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"] = "a0fc35f3a5044534a65010f646172a48"
# os.environ["COMPUTER_VISION_ENDPOINT"] = "https://cv-rambam-test.cognitiveservices.azure.com/"
def AnalyzeMeasures(frame):
    COMPUTER_VISION_ENDPOINT = os.environ["COMPUTER_VISION_ENDPOINT"]
    COMPUTER_VISION_SUBSCRIPTION_KEY = os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"]

    # Creating client instance:
    computervision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY))

    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(frame), raw=True)
    # recognize_printed_results = computervision_client.batch_read_file_in_stream((frame), raw=True)

    # Reading OCR results
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
                break
        time.sleep(0.5)

    data_dict = {}
    i = 1
    # tmp_frame = cv2.imencode(".jpg", frame)[1]
    tmp_frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
                for line in text_result.lines:
                        # print("Measure: ", line.text, " | Sum xyz: ", sum(line.bounding_box))
                        print(line.bounding_box)
                        data_dict[i] = re.sub('[^0123456789./]', '', line.text)
                        cv2.rectangle(tmp_frame, (int(line.bounding_box[0]), int(line.bounding_box[1])), (int(line.bounding_box[4]), int(line.bounding_box[5])), (255,0,0), 2)
                        cv2.imshow("image", tmp_frame)
                        cv2.waitKey(0)
                        i = i + 1
    data_json = json.dumps(data_dict)
    data_string = "\"" + data_json + "\""
    headers = {'Content-type': 'string'}
    print(data_string)
    # response = requests.post(url, data=data_string, headers=headers)

    # TODO: sanity check results (charecters etc.) and send them to somewhere
    return
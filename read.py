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

# Setting enviroment variables:
#TODO: incorporate along the way:
os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"] = "a0fc35f3a5044534a65010f646172a48"
os.environ["COMPUTER_VISION_ENDPOINT"] = "https://cv-rambam-test.cognitiveservices.azure.com/"

COMPUTER_VISION_ENDPOINT = os.environ["COMPUTER_VISION_ENDPOINT"]
COMPUTER_VISION_SUBSCRIPTION_KEY = os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"]

# Creating client instance:
computervision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY))
"""
# Image path: 
img_path = "https://i.ibb.co/6WjvmFq/try-Copy-2.jpg"
# Image OCR request:
recognize_printed_results = computervision_client.batch_read_file(img_path, raw=True)
"""
imageStream = open("/data/home/avihay/VideoAnalytics/try.jpg", "rb")
recognize_printed_results = computervision_client.batch_read_file_in_stream(imageStream, raw=True)


# Reading OCR results
operation_location_remote = recognize_printed_results.headers["Operation-Location"]
operation_id = operation_location_remote.split("/")[-1]
while True:
    get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
    if get_printed_text_results.status not in ['NotStarted', 'Running']:
            break
    time.sleep(1)

if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
    for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
                    print(line.text)
                    print(line.bounding_box)
# TODO: sanity check results (charecters etc.) and send them to somewhere

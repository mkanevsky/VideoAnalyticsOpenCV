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
import math
import numpy as np
import base64


def create_bounded_output(readings, boundings, boundries):
    output_dict = {}
    for key in boundries.keys():
        for i in range(len(readings)):
            if check_boundry(boundings[i], boundries[key]):
                output_dict[key] = readings[i]
        if key not in output_dict.keys():
            output_dict[key] = "N/A"
    return output_dict

def check_boundry(bounding, boundry):
    output = bounding[0]>=boundry[0]
    output = output and (bounding[6]>=boundry[0])
    output = output and (bounding[2]<=boundry[1])
    output = output and (bounding[4]<=boundry[1])
    output = output and (bounding[1]>=boundry[2])
    output = output and (bounding[3]>=boundry[2])
    output = output and (bounding[5]<=boundry[3])
    output = output and (bounding[7]<=boundry[3])
    return output 
"""
def create_bounded_output(readings, boundings, boundries, x, y):
    output_dict = {}
    for key in boundries.keys():
        for i in range(len(readings)):
            if check_boundry(boundings[i], boundries[key], x, y):
                output_dict[key] = readings[i]
        if key not in output_dict.keys():
            output_dict[key] = "N/A"
    return output_dict

def check_boundry(bounding, boundry,x ,y):
    output = bounding[0]>=boundry[0]*x
    output = output and (bounding[6]>=boundry[0]*x)
    output = output and (bounding[2]<=boundry[1]*x)
    output = output and (bounding[4]<=boundry[1]*x)
    output = output and (bounding[1]>=boundry[2]*y)
    output = output and (bounding[3]>=boundry[2]*y)
    output = output and (bounding[5]<=boundry[3]*y)
    output = output and (bounding[7]<=boundry[3]*y)
    return output  
"""
def fix_string(s):
    json_string_fin = ""
    last_c=""
    for c in s:
        if c!="\'":
            json_string_fin += c
            if c=="{":
                if last_c=="\'":
                    json_string_fin = last_string + "\'{"
        else:
            last_string = json_string_fin
            if last_c!="}":
                json_string_fin += "\""
            else:
                json_string_fin += "\'"
        last_c = c
    return json_string_fin

def bounding_boxes_output_former(bbox_dict, mon_id, encoded_image):
    string_json = json.dumps(bbox_dict)
    json_dict = {}
    json_dict["JsonData"] = string_json
    json_dict["MonitorID"] = mon_id
    json_dict["MonitorImage"] = encoded_image
    json_dict_string = str(json_dict)
    print(json_dict_string)
    output = fix_string(json_dict_string)
    return output


def get_digits(img, computervision_client):
    # encodedFrame = cv2.imencode(".jpg", img)[1].tostring()
    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(img), raw = True)
    # Reading OCR results
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
                break
        time.sleep(0.1)
    
    tmp_frame = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    results = []
    text_flag = True
    show_frame_flag = True
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
                # print(line.text, line.bounding_box)
                s = re.sub('[^0123456789./]', '', line.text)
                if s != "":
                    if s[0] == ".":
                        s = s[1:]
                    s = s.rstrip(".")
                    text_flag = True
                    top_left_coords = (int(line.bounding_box[0]), int(line.bounding_box[1]))
                    bottom_right_coords = (int(line.bounding_box[4]), int(line.bounding_box[5]))
                    # cv2.rectangle(tmp_frame, (int(line.bounding_box[0]), int(line.bounding_box[1])), (int(line.bounding_box[4]), int(line.bounding_box[5])), (255,0,0), 2)
                    cv2.rectangle(tmp_frame, top_left_coords, bottom_right_coords, (255,0,0), 2)
                    # cv2.putText(tmp_frame,s,(int(line.bounding_box[0])-5, int(line.bounding_box[1])-5),cv2.FONT_HERSHEY_COMPLEX,0.3,(0,0,0),1)
                    # results.append((s, line.bounding_box))
                    results.append((top_left_coords, bottom_right_coords))
                else:
                    continue
        if text_flag and show_frame_flag:
            print(results)
            cv2.imshow("image", tmp_frame)
            cv2.waitKey(0)
    return(results)

#TODO: incorporate along the way:

def AnalyzeFrame(frame, computervision_client):
    # COMPUTER_VISION_ENDPOINT = os.environ["COMPUTER_VISION_ENDPOINT"]
    # COMPUTER_VISION_SUBSCRIPTION_KEY = os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"]
    # #TODO: get client as argument
    # computervision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY))
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    s = frame.shape
    # tmp = cv2.imencode(".jpg", frame)[1]
    # cv2.imwrite("try.jpg", tmp)
    frame_height, frame_width = s[0], s[1]
    # w1 = 0.7
    # w2 = 0.9
    # h1 = 0
    # h2 = frame_height
    crop_img_side = frame[0:frame_height, math.ceil(0.7*frame_width):math.ceil(0.90*frame_width)]
    crop_img_low = frame[math.ceil(0.7*frame_height):frame_height, math.ceil(0.3*frame_width):math.ceil(0.7*frame_width)]
    # areas = [crop_img_side, crop_img_low]
    areas = [crop_img_side]

    # our output
    coords = {}
    i = 0
    for img in areas:
        # print(type(img))
        result = get_digits(cv2.imencode(".jpg", img)[1], computervision_client)
        for item in result:
            coords[i] = item
            i = i + 1
    print(coords)
    transformed_coords = {k: tuple((int(x[0]+0.7*frame_width), x[1]) for x in v) for k,v in coords.items()}
    print(transformed_coords)

    b64img = base64.b64encode(cv2.imencode(".jpg", frame)[1])

    b64_encoded_frame = b64img.decode('utf-8')
    # Check transformed coords:
    # tmp_frame = frame
    # for k, v in transformed_coords.items():
    #     cv2.rectangle(tmp_frame, v[0], v[1], (255,0,0), 2)
    # cv2.imshow("image", tmp_frame)
    # cv2.waitKey(0)
    
    monitor_id = "1"
    json_string_fin = bounding_boxes_output_former(transformed_coords, monitor_id, b64_encoded_frame)
    # print(json_string_fin)
    # print("--- %s seconds ---" % (time.time() - start_time))
    url = "http://rstreamapp.azurewebsites.net/api/UploadMonitorMapping"
    headers={'Content-type':'application/json', 'Accept':'application/json'}
    response = requests.post(url, data=json_string_fin, headers=headers)
    # print(response)

    #print('results for frame: ', result_list)

     

    # TODO: sanity check results (charecters etc.) and send them to somewhere
    return



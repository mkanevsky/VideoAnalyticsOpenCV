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

def output_former(ocr_res, room, pat_id, mon_id):
    string_json = json.dumps(ocr_res)
    json_dict = {}
    json_dict["JsonData"] = string_json
    json_dict["MonitorID"] = mon_id
    json_dict["PatientID"] = pat_id
    json_dict["Room"] = room
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

    # string_dict = {}
    # i = 1
    # if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
    #     for text_result in get_printed_text_results.recognition_results:
    #             for line in text_result.lines:
    #             #        print(line.text)
    #             #       print(line.bounding_box)
    #                     s = re.sub('[^0123456789./]', '', line.text)
    #                     if s != "":
    #                         if s[0] == ".":
    #                             s = s[1:]
    #                         s = s.rstrip(".")
    #                     else:
    #                         continue
    #                     string_dict[i] = s
    #                     i += 1
    
    tmp_frame = cv2.imdecode(np.frombuffer(img, np.uint8), -1)
    results = []
    text_flag = False
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
                    cv2.rectangle(tmp_frame, (int(line.bounding_box[0]), int(line.bounding_box[1])), (int(line.bounding_box[4]), int(line.bounding_box[5])), (255,0,0), 2)
                    results.append((s, line.bounding_box))
                else:
                    continue
        if text_flag and show_frame_flag:
            cv2.imshow("image", tmp_frame)
            cv2.waitKey(0)
    return(results)
import cv2
import math
import numpy as np
#TODO: incorporate along the way:
# os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"] = "a0fc35f3a5044534a65010f646172a48"
# os.environ["COMPUTER_VISION_ENDPOINT"] = "https://cv-rambam-test.cognitiveservices.azure.com/"
def AnalyzeFrame(frame):
    COMPUTER_VISION_ENDPOINT = os.environ["COMPUTER_VISION_ENDPOINT"]
    COMPUTER_VISION_SUBSCRIPTION_KEY = os.environ["COMPUTER_VISION_SUBSCRIPTION_KEY"]
    #TODO: get client as argument
    computervision_client = ComputerVisionClient(COMPUTER_VISION_ENDPOINT, CognitiveServicesCredentials(COMPUTER_VISION_SUBSCRIPTION_KEY))
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    s =frame.shape
    # tmp = cv2.imencode(".jpg", frame)[1]
    # cv2.imwrite("try.jpg", tmp)
    y,x = s[0], s[1]
    crop_img_side = frame[0:y, math.ceil(0.70*x):math.ceil(0.90*x)]
    crop_img_low = frame[math.ceil(0.7*y):y, math.ceil(0.3*x):math.ceil(0.7*x)]
    areas = [crop_img_side, crop_img_low]

    # our output
    readings = {}
    boundings = {}
    i = 0
    for img in areas:
        # print(type(img))
        results = get_digits(cv2.imencode(".jpg", img)[1], computervision_client)
        for item in results:
            readings[i] = item[0]
            boundings[i] = item[1]
            i = i + 1
    
    boundry_dict = {i:[min(x[0],x[6]) -15,max(x[2],x[4]) +15 ,min(x[3],x[1]) -15,max(x[5],x[7]) + 15] for i,x in enumerate(boundings.values())}
    # print(boundry_dict)
    boundry_temp2 = {0: [10.0, 46.0, 38.0, 68.0], 1: [8.0, 67.0, 62.0, 94.0], 2: [9.0, 55.0, 83.0, 108.0], 3: [5.0, 36.0, 107.0, 132.0], 4: [7.0, 34.0, 128.0, 152.0]}
    boundry_temp3 = {0: [-11.0, 50.0, 32.0, 85.0], 1: [-12.0, 62.0, 56.0, 106.0], 2: [-9.0, 60.0, 77.0, 124.0], 3: [-9.0, 45.0, 99.0, 147.0], 4: [-10.0, 37.0, 121.0, 166.0]}
    boundry_temp = {0: [0.04656862745098039, 0.12826797385620914, 0.20147420147420148, 0.29975429975429974], 1: [0.03594771241830065, 0.1968954248366013, 0.3046683046683047, 0.414004914004914], 2: [0.0457516339869281, 0.15522875816993464, 0.41154791154791154, 0.47911547911547914], 3: [0.03104575163398693, 0.10294117647058823, 0.5147420147420148, 0.6130221130221131], 4: [0.03594771241830065, 0.09803921568627451, 0.6130221130221131, 0.7014742014742015], 5: [0.07598039215686274, 0.1772875816993464, -0.002457002457002457, 0.09705159705159705]}
    output = create_bounded_output(readings, boundings, boundry_temp3)
    # print(output)

    pat_id = "200465524"
    room = "13"
    mon_id = "90210"
    # start_time = time.time()
    json_string_fin = output_former(output, room, pat_id, mon_id)
    print(json_string_fin)
    # print("--- %s seconds ---" % (time.time() - start_time))
    url = "http://rstreamapp.azurewebsites.net/api/InsertMonitorData"
    headers={'Content-type':'application/json', 'Accept':'application/json'}
    response = requests.post(url, data=json_string_fin, headers=headers)

    #print('results for frame: ', result_list)

    # Creating client instance:
    

    """ CURRENT CODE: """
    """

    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(frame), raw=True)
    # recognize_printed_results = computervision_client.batch_read_file_in_stream((frame), raw=True)

    
    
    # Reading OCR results
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
                break
        time.sleep(0.05)

    
    string_dict = {}
    i = 1
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
                for line in text_result.lines:
                #        print(line.text)
                #       print(line.bounding_box)
                        s = re.sub('[^0123456789./]', '', line.text)
                        if s != "":
                            if s[0] == ".":
                                s = s[1:]
                            s = s.rstrip(".")
                        else:
                            continue
                        string_dict[i] = s
                        i += 1

    pat_id = "200465524"
    room = "13"
    mon_id = "90210"
    # start_time = time.time()
    json_string_fin = output_former(string_dict, room, pat_id, mon_id)
    print(json_string_fin)
    # print("--- %s seconds ---" % (time.time() - start_time))
    url = "http://rstreamapp.azurewebsites.net/api/InsertMonitorData"
    headers={'Content-type':'application/json', 'Accept':'application/json'}
    response = requests.post(url, data=json_string_fin, headers=headers)
    # print(" Reponse::   ", response)

    # url = "http://rstreamapp.azurewebsites.net/api/GetMonitorData?PatientID=200465524&MonitorID=1"
    # params = {"format": "json"}
    # r = requests.get(url, params= params) 
    # print(r)
    """

    """
    data_dict = {}
    i = 1
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
                for line in text_result.lines:
                        # print("Measure: ", line.text, " | Sum xyz: ", sum(line.bounding_box))
                        # print(sum(line.bounding_box))
                        data_dict[i] = re.sub('[^0123456789./]', '', line.text)
                        i = i + 1
    
    pat_id = "200465524"
    room = "13"
    mon_id = "90210"
    json_string_fin = 
    
    data_json = json.dumps(data_dict)
    data_string = "\"" + data_json + "\""
    headers = {'Content-type':'application/json', 'Accept':'application/json'}
    print(data_string)
    url = "http://rstreamapp.azurewebsites.net/api/InsertMonitorData"
    response = requests.post(url, data=data_string, headers=headers)
    """
    

    # TODO: sanity check results (charecters etc.) and send them to somewhere
    return



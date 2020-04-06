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


def sliding_window(image, step_size, window_size):
	# slide a window across the image
  for y in range(0, image.shape[0], step_size):
    for x in range(0, image.shape[1], step_size):
			# yield the current window
      if (y + window_size[1] <= image.shape[0]) and (x + window_size[0] <= image.shape[1]):
        yield (x, y, image[y:y + window_size[1], x:x + window_size[0]]) #we want only full windows
      else:
        continue


def find_best_windows(computervision_client, warped_frame, num_of_windows=1):
    best_score_v = 0
    best_window_v = []
    best_window_h = []
    s = warped_frame.shape
    # TODO: create larger windows (all the widht/length of the frame)
    # define vertical sliding window size to be 0.25X by Y
    winH, winW = s[0], math.ceil(0.25*s[1])
    step_size = math.ceil(s[1] / 10)

    # Pre Process:
    
    gray = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    warped_frame = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, gray)
    

    min_size = (s[0]/100) * (s[1]/100)  # min_size is set to be 1X1% of monitor
    for (x, y, window) in sliding_window(warped_frame, step_size, window_size=(winW, winH)):  # vertical windows
      temp_results = get_digits_FBW(window, computervision_client)
      # filter all results smaller than min size
      temp_results = [x for x in temp_results if (
          (x[1][4] - x[1][0]) * (x[1][5] - x[1][1])) > min_size]
      temp_score = len(temp_results)
      if temp_score > best_score_v:
        best_score_v = temp_score
        # [x, x + winW, y, y + winH]
        best_window_v = [x, x + winW, y, y + winH]
    if num_of_windows == 2:
      # define horizontal sliding window size to be 0.3Y by X
      winH, winW = math.ceil(0.3*s[0]), s[1]
      step_size = math.ceil(s[0] / 10)
      best_score_h = 0
      processed_frame_h = cv2.rectangle(warped_frame, (best_window_v[0], best_window_v[2]), (
          best_window_v[1], best_window_v[3]), (0, 255, 0), -1)  # block best vertical area
      for (x, y, window) in sliding_window(processed_frame_h, step_size, window_size=(winW, winH)):  # horizontal windows
        temp_results = get_digits_FBW(window, computervision_client)
        # filter all results smaller than min size
        temp_results = [x for x in temp_results if (
            (x[1][4] - x[1][0]) * (x[1][5] - x[1][1])) > min_size]
        temp_score = len(temp_results)
        if temp_score > best_score_h:
          best_score_h = temp_score
          best_window_h = [x, x + winW, y, y + winH]
    # areas_dict value format is: [y_down, y_up, x_left, x_right]
    final_result = []
    final_result.append([best_window_v[2] / s[0], best_window_v[3] /
                         s[0], best_window_v[0] / s[1], best_window_v[1] / s[1]])
    if best_window_h:
        final_result.append([best_window_h[2] / s[0], best_window_h[3] /
                             s[0], best_window_h[0] / s[1], best_window_h[1] / s[1]])
    return final_result


def create_areas(area_dict, img):
    s = img.shape
    height, width = s[0], s[1]
    areas = []
    for key, value in area_dict.items():
        hmin, hmax, wmin, wmax = value
        # print(hmin, hmax, wmin, wmax)
        hmin *= height
        hmax *= height
        wmin *= width
        wmax *= width
        # print(hmin, hmax, wmin, wmax)
        new_area = [img[math.ceil(hmin):math.ceil(hmax), math.ceil(wmin):math.ceil(wmax)], hmin, wmin]
        areas.append(new_area)
    return areas

def transform_coords(coords, area, mode='avihay'):
    # print(coords[0][0])
    topleft = (coords[0][0]+area[2], coords[0][1]+area[1])
    bottomright = (coords[1][0]+area[2], coords[1][1]+area[1])
    return (topleft, bottomright)


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
    # print(json_dict_string)
    output = fix_string(json_dict_string)
    return output

def get_digits_FBW(image, computervision_client):
    encodedFrame = cv2.imencode(".jpg", image)[1].tostring()
    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(encodedFrame), raw = True)
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
            break
    results = []
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
              s = re.sub('[^0123456789./]', '', line.text)
              if s != "":
                  if s[0] == ".":
                      s = s[1:]
                  s = s.rstrip(".")
                  results.append((s, line.bounding_box))
              else:
                  continue
    return results


def get_digits(img, computervision_client):
    encodedFrame = cv2.imencode(".jpg", img)[1].tostring()
    recognize_printed_results = computervision_client.batch_read_file_in_stream(io.BytesIO(encodedFrame), raw = True)
    # Reading OCR results
    operation_location_remote = recognize_printed_results.headers["Operation-Location"]
    operation_id = operation_location_remote.split("/")[-1]
    while True:
        get_printed_text_results = computervision_client.get_read_operation_result(operation_id)
        if get_printed_text_results.status not in ['NotStarted', 'Running']:
                break
        time.sleep(0.1)
    
    tmp_frame = cv2.imdecode(np.frombuffer(encodedFrame, np.uint8), -1)
    results = []
    text_flag = True
    show_frame_flag = False
    if get_printed_text_results.status == TextOperationStatusCodes.succeeded:
        for text_result in get_printed_text_results.recognition_results:
            for line in text_result.lines:
                print(line.text, line.bounding_box)
                s = re.sub('[^0123456789./]', '', line.text)
                if s != "":
                    if s[0] == ".":
                        s = s[1:]
                    s = s.rstrip(".")
                    text_flag = True
                    top_left_coords = (int(line.bounding_box[0]), int(line.bounding_box[1]))
                    bottom_right_coords = (int(line.bounding_box[4]), int(line.bounding_box[5]))
                    cv2.rectangle(tmp_frame, top_left_coords, bottom_right_coords, (255,0,0), 2)
                    results.append((top_left_coords, bottom_right_coords))
                else:
                    continue
        if text_flag and show_frame_flag:
            print(results)
            cv2.imshow("image", tmp_frame)
            cv2.waitKey(0)
    return(results)


def AnalyzeMeasures(frame, computervision_client):
    frame = cv2.imdecode(np.frombuffer(frame, np.uint8), -1)
    # cv2.imwrite("image.jpg", frame)
    """
    # areas_dict = {'side': [0, 1, 0.7, 0.9], 'bottom': [0.6, 0.9, 0.3, 0.7]}
    areas_dict = {'low': [0.6, 0.85, 0, 0.5], 'side': [0.1, 0.9, 0.6, 0.9]}
    areas = create_areas(areas_dict, frame)
    """
    # print(type(frame))

    areas_of_intrest = find_best_windows(computervision_client, frame, 2) #find best windows
    areas_dict = {i:area for i,area in enumerate(areas_of_intrest)} #transform into dictionary of bounderies
    areas = create_areas(areas_dict, frame)


    # our output
    coords = {}
    transformed_coords = {}
    i = 0
    for area in areas:
        result = get_digits(area[0], computervision_client)
        print(result)
        for item in result:
            print(i, item)
            coords[i] = item
            transformed_coords[i] = transform_coords(item, area)
            i = i + 1
    print("fixed coords are:", transformed_coords)
    #TODO: add argument to choose whether or not to send response (send and/or print)
    return

    b64img = base64.b64encode(cv2.imencode(".jpg", frame)[1])
    b64_encoded_frame = b64img.decode('utf-8')
    
    #TODO: get this data from Shany's DB
    monitor_id = "90210"
    json_string_fin = bounding_boxes_output_former(transformed_coords, monitor_id, b64_encoded_frame)
    # print(json_string_fin)
    url = "http://rstreamapp.azurewebsites.net/api/UploadMonitorMapping"
    headers = {'Content-type':'application/json', 'Accept':'application/json'}
    for _ in range(4):
        while True:
            try:
                response = requests.post(url, data=json_string_fin, headers=headers)
            except Exception as e:
                print("Exception while posting:   |    ", e)
                # TODO: throw exception if all trails ended unsuccefuly
                continue
            break
    #print(response)     

    # TODO: sanity check results (charecters etc.) and send them to somewhere
    return



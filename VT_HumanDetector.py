import socket
import time
import cv2
import mediapipe as mp
import numpy as np
from enum import Enum
import json
from datetime import datetime

class FaceKeyPoints(Enum):
    NOSE_CENTER = 1,        #1
    JAW＿CENTER = 152,
    JAW_LEFT = 378,
    JAW_RIGHT = 149,
    EYE_RIGHT_OUT = 33,     #5
    EYE_RIGHT_IN = 133,
    EYE_RIGHT_TOP = 159,
    EYE_RIGHT_BTM = 145,
    EYE_LEFT_OUT = 263,
    EYE_LEFT_IN = 362,      #10
    EYE_LEFT_TOP = 386,
    EYE_LEFT_BTM = 374,
    MOUTH_TOP = 12,
    MOUTH_UL = 271,
    MOUTH_LEFT = 292,       #15
    MOUTH_LB = 403,
    MOUTH_BTM = 15,
    MOUTH_RB = 179,
    MOUTH_RIGHT = 62,
    MOUTH_RT = 14,          #20
    FORE_HEAD_CENTER = 10,
    FACE_RIGHT = 234,
    FACE_LEFT = 454,

class DataSender:
    def __init__(self, host_ip, host_port):
        self.host = host_ip
        self.port = host_port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        pass

    def send(self, data_dic):
        json_data_str = json.dumps(data_dic)
        self.client.sendto(json_data_str.encode('utf-8'), (self.host, self.port))
        return


def main():
    data_sender = DataSender('192.168.11.5', 56123)
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    send_data = False

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while(cap.isOpened()):
            success, image = cap.read() 
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            mp_drawing.draw_landmarks(
                image,
                results.face_landmarks,
                mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles
                .get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles
                .get_default_pose_landmarks_style())
            cv2.imshow('MediaPipe Holistic', image)

            key = cv2.waitKey(5)
            if key & 0xFF == 27:
                break

            if key & 0xFF == 32:
                send_data = True
            
            if send_data:
                json_data = createJsonDataDictionary(results.face_landmarks, results.pose_landmarks, image)
                data_sender.send(json_data)  
                pass


    cap.release()
    cv2.destroyAllWindows()


def createJsonDataDictionary(face_landmarks, body_randmarks, img):
    json_data = {
        "Body": [],
        "Face" : [],
        "R_hand" : [],
        "L_hand" : [],
        "Timestamp": "",
        "ImHeight" : 0,
        "ImWidth" : 0,
    }

    if face_landmarks != None:
        keypoints = []
        for face_kp in FaceKeyPoints:
            point = face_landmarks.landmark[face_kp.value[0]]
            keypoints.append({
            "x": point.x,
            "y": point.y,
            "z": point.z,
            "visibility": point.visibility,})
        json_data["Face"] = keypoints

    json_data["Timestamp"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    json_data["ImHeight"] = img.shape[0]
    json_data["ImWidth"] = img.shape[1]

    return json_data

if __name__ == "__main__":
    main()

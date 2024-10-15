import streamlit as st
from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import dlib
import cv2
import simpleaudio as sa
import threading
import base64
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Load the facial landmark predictor
svm_predictor_path = 'SVMclassifier.dat'
EYE_AR_THRESH = 0.20
EYE_AR_CONSEC_FRAMES = 10
MOU_AR_THRESH = 1.10   

# Alarm function
def play_alarm(file_path):
    with open(file_path, "rb") as f:
            audio_data = f.read()
    b64_audio = base64.b64encode(audio_data).decode()
    return f"""
    <audio autoplay>
        <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
        Your browser does not support the audio element.
    </audio>
    """
# EAR and MOR calculation functions
def EAR(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def MAR(mouth):
    A = dist.euclidean(mouth[0], mouth[6])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    return (B + C) / A

# Initialize Dlib’s face detector and facial landmark predictor
svm_detector = dlib.get_frontal_face_detector()
svm_predictor = dlib.shape_predictor(svm_predictor_path)
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.COUNTER = 0
        self.alarm_on = False
        self.yawn_status = False
        self.yawns = 0

    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = svm_detector(gray, 0)

        for rect in rects:
            shape = svm_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]

            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = MAR(mouth)

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)

            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

            if ear < EYE_AR_THRESH:
                self.COUNTER += 1
                cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.COUNTER >= EYE_AR_CONSEC_FRAMES and not self.alarm_on:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.alarm_on = True
                    st.markdown(play_alarm("alarm.wav"), unsafe_allow_html=True)
                    self.alarm_on = False
            else:
                self.COUNTER = 0
                self.alarm_on = False
                cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            if mar > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning, DROWSINESS ALERT!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                self.yawn_status = True
                output_text = "Yawn Count: " + str(self.yawns + 1)
                cv2.putText(frame, output_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                if not self.alarm_on:
                    self.alarm_on = True
                    st.markdown(play_alarm("alarm.wav"), unsafe_allow_html=True)
                    self.alarm_on = False
            else:
                self.yawn_status = False

            if self.yawn_status and not self.yawn_status:
                self.yawns += 1
                
            cv2.putText(frame, "MAR: {:.2f}".format(mar), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

st.title("Driver Drowsiness Monitoring System")
st.write("Real-time monitoring using Visual Behaviour and Machine Learning")
webrtc_streamer(
    key="drowsiness_detection",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=VideoTransformer,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
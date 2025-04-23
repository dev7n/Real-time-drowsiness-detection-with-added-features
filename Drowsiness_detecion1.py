from __future__ import division
import dlib
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import numpy as np
from scipy.spatial import distance as dist
import threading
import pygame
import time

# Initialize sound
def start_sound():
    pygame.mixer.init()
    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()

# Compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Compute Ambient Light Level
def get_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

# Apply Night Mode Enhancement
def apply_night_mode(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced_frame

# Head Pose Estimation
def get_head_pose(shape):
    image_points = np.array([
        (shape[30][0], shape[30][1]),  # Nose tip
        (shape[8][0], shape[8][1]),    # Chin
        (shape[36][0], shape[36][1]),  # Left eye left corner
        (shape[45][0], shape[45][1]),  # Right eye right corner
        (shape[48][0], shape[48][1]),  # Left Mouth corner
        (shape[54][0], shape[54][1])   # Right mouth corner
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -330.0, -65.0),   # Chin
        (-225.0, 170.0, -135.0), # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    focal_length = 1 * 640
    center = (640 / 2, 480 / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
    return rotation_vector

# Initialize camera and Dlib face detector
camera = cv2.VideoCapture(0)
predictor_path = 'shape_predictor_68_face_landmarks.dat_2'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
(lStart, lEnd) = (42, 48)  # Right Eye
(rStart, rEnd) = (36, 42)  # Left Eye

# Drowsiness Detection Loop
print("[INFO] Drowsiness Detection Started...")
total = 0
alarm = False

while True:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    brightness = get_brightness(frame)
    
    # Apply night mode if brightness is low
    if brightness < 70:
        frame = apply_night_mode(frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)])
        leftEAR = eye_aspect_ratio(shape[lStart:lEnd])
        rightEAR = eye_aspect_ratio(shape[rStart:rEnd])
        ear = (leftEAR + rightEAR) / 2.0
        rotation_vector = get_head_pose(shape)
        head_tilt = rotation_vector[0][0] * 100

        if ear < 0.22 or head_tilt > 1:
            total += 1
            if total > 20:
                if not alarm:
                    alarm = True
                    threading.Thread(target=start_sound, daemon=True).start()
                    cv2.putText(frame, "DROWSINESS DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            cv2.putText(frame, f"Eyes Closed ({total})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            total = 0
            alarm = False
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Head Tilt: {head_tilt:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    cv2.putText(frame, f"Brightness: {brightness:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Drowsiness Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()


import argparse
import pathlib
import numpy as np
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps

from batch_face import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render
#import mediapipe as mp
#import pygame 



cap = cv2.VideoCapture(0)

cudnn.enabled = True
device = 'gpu:0' # 'cpu'
CWD = pathlib.Path.cwd()
gaze_pipeline = Pipeline(
    weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
    arch='ResNet50', # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
    device = select_device(device, batch_size=1)
)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

print(torch.cuda.is_available()) # not available yet


#def parse_args():
#    """Parse input arguments."""
#    parser = argparse.ArgumentParser(
#        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
#    parser.add_argument(
#        '--device',dest='device', help='Device to run model: cpu or gpu:0',
#        default="cpu", type=str)
#    parser.add_argument(
#        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
#        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
#    parser.add_argument(
#        '--cam',dest='cam_id', help='Camera device id to use [0]',  
#        default=0, type=int)
#    parser.add_argument(
#        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
#        default='ResNet50', type=str)
#
#    args = parser.parse_args()
#    return args

#def alert_on():
#    print("Wake up!")
    
#def alert_off():
#    print("I am glad you woke up. Please, get some sleep")


# returns one frame from camera and turns it off
def get_frame():
    _, frame = cap.read()
    return frame
    

#Get the frame and vector pos
def get_gaze():
    frame = get_frame()
    with torch.no_grad(): 
        results = gaze_pipeline.step(frame)
        return results, frame
    

#Define the normal field of view
def calibration():
    results = []
    texts = ["1. look at the top left corner", "2. look at the top right corner", 
             "3. look at the bottom right corner", "4. look at the bottom left corner"]
    for i in range(0, 4):
        input(texts[i])
        gg, frm = get_gaze()
        print(type(frm))
        results.append(gg)    
        #cv2.imshow("window", frm)
         
    min_pitch = min(gg.pitch for gg in results)
    max_pitch = max(gg.pitch for gg in results)
    min_yaw = min(gg.yaw for gg in results)
    max_yaw = max(gg.yaw for gg in results)
    return min_pitch, max_pitch, min_yaw, max_yaw
    

img = np.zeros( (1080,1920,3), np.uint8)
cv2.namedWindow("window", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("window",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
#cv2.resizeWindow("window", 1920, 1080)
#cv2.imshow('window', img)


#cv2.putText(img, '1920,0', (1920,0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#cv2.putText(img, '1920,1080', (1920,1080), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#cv2.putText(img, '0,1080', (0,1080), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#cv2.putText(img, '0,0', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



# Check if person looks within calibrated limits
def check_gaze(pitch, yaw, min_pitch, max_pitch, min_yaw, max_yaw):
    
    x = (yaw - min_yaw) / (max_yaw - min_yaw) * 1920
    y = (pitch - min_pitch) / (max_pitch - min_pitch) * 1080
    x = int(x) # 1920 - ???
    y = int(y) # 1080 - int(y) ????
    
    #TODO something wrong with x,y or CV coord system
    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(img, str(x)+','+str(y), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow('window', img)
    key = cv2.waitKey(10)
    
    # TODO:
    #  rebuild env with the addition of:
    #   pip install pyautogui
    #   gpu cuda support
    
    #  currently the accuracy of x and y is somewhat bad
    #  if you can train image model to predict x,y it would be more accurate I guess
    
    if (min_pitch < pitch < max_pitch) and (min_yaw < yaw < max_yaw):
        print(x,y)
        return True
    else:
        return False
    
    
    
  
if __name__ == '__main__':
    #args = parse_args()
    #cam = args.cam_id
    #snapshot_path = args.snapshot
    
    # cv2.waitKey(5000) 
    
    min_pitch, max_pitch, min_yaw, max_yaw = calibration()
    print(min_pitch, max_pitch, min_yaw, max_yaw)

    
    while True:
        #input()
        results, frm = get_gaze()
        cv2.imshow("window", frm)
        l = check_gaze(results.pitch, results.yaw, min_pitch, max_pitch, min_yaw, max_yaw)
        #print(l)
    
    """
    # Check if the webcam is opened correctly
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

    # Define landmarks for left and right eyes (based on Mediapipe's FaceMesh)
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def calculate_ear(eye_landmarks):        
        # Calculate the Eye Aspect Ratio (EAR).
        # EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)

        vertical_1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        vertical_2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        horizontal = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear

    # Threshold for determining if eyes are closed
    EAR_THRESHOLD = 0.25
    CLOSED_EYE_TIME_THRESHOLD = 1.5  # Time in seconds

    # Initialize pygame mixer for sound playback
    #pygame.mixer.init()
    #pygame.mixer.music.load("alarm.mp3")  # Replace with the path to your alert sound file


    # Timer variables
    eyes_closed_start = None
    alarm_triggered = False
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert to RGB for Mediapipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract eye landmarks
                    left_eye = [(face_landmarks.landmark[i].x * frame.shape[1],
                                face_landmarks.landmark[i].y * frame.shape[0]) for i in LEFT_EYE]
                    right_eye = [(face_landmarks.landmark[i].x * frame.shape[1],
                                face_landmarks.landmark[i].y * frame.shape[0]) for i in RIGHT_EYE]

                    # Calculate EAR for both eyes
                    left_ear = calculate_ear(left_eye)
                    right_ear = calculate_ear(right_eye)

                    # Determine if eyes are closed
                    if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                        if eyes_closed_start is None:
                            eyes_closed_start = time.time()  # Start timer
                        else:
                            closed_duration = time.time() - eyes_closed_start
                            if closed_duration > CLOSED_EYE_TIME_THRESHOLD and not alarm_triggered:
                                print("ALERT: Eyes closed for too long!")
                                #pygame.mixer.music.play(loops=-1)  # Play alert sound in a loop
                                alert_on()
                                alarm_triggered = True
                                cv2.putText(frame, f"Eyes Closed ({int(closed_duration)}s)", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        eyes_closed_start = None  # Reset timer
                        if alarm_triggered:
                            #pygame.mixer.music.stop()  # Stop the alert sound
                            alarm_triggered = False
                        cv2.putText(frame, "Eyes Open", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    # Draw eye landmarks
                    for point in left_eye + right_eye:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 2, (255, 0, 0), -1)

            cv2.imshow("Eye Tracking", frame)

            # Get frame
            success, frame = cap.read()    
            start_fps = time.time()  



            # Visualize output
            frame = render(frame, results)
           
            myFPS = 1.0 / (time.time() - start_fps)
            #cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, 'Pays attention: '+str(l), (10, 20),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("Demo",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            success,frame = cap.read()  
"""



cap.release()

cv2.destroyAllWindows()
 
    

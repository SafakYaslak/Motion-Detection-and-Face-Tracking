import numpy as np
import cv2
import time
import datetime as datetime
import mediapipe as mp
import os
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture("Undetected_Video.mp4")


MOTİON_DETECT = "Captured Motion"
DETECET_FACES = "If Any Faces"
MERGED= "Merged"
def draw_hsv(flow):
    h,w = flow.shape[:2]       
    fx,fy = flow[:,:,0] , flow[:,:,1]

    ang = np.arctan2(fy,fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv= np.zeros((h,w,3),np.uint8)
    hsv[...,0]=ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4,255)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr

def draw_flow(img, flow, step=16):
    h,w = img.shape[:2] 

    y, x=  np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int) ## gridlerin konumlarını hesaplamak için 
    fx,fy = flow[y,x].T
    lines=np.vstack([x,y,x-fx,y-fy]).T.reshape(-1,2,2)
    lines = np.int32(lines +0.5)

    img_bgr = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr,lines,0,(0,255,0))
    for (x1,y1),(_x2,_y2) in lines:
        cv2.circle(img_bgr,(x1,y1),1,(0,255,0),-1)
    return img_bgr

def face_rectangle_coordiantes(image, face_landmarks):
    min_x, max_x = float('inf'), 0
    min_y, max_y = float('inf'), 0

    for landmark in face_landmarks.landmark:
        x, y, z = landmark.x, landmark.y, landmark.z
        image_height, image_width, _ = image.shape
        x_px, y_px = int(x * image_width), int(y * image_height)

        # Minimum ve maksimum x ve y koordinatlarını güncelle
        min_x = min(min_x, x_px)
        max_x = max(max_x, x_px)
        min_y = min(min_y, y_px)
        max_y = max(max_y, y_px)

    # Dikdörtgen çizme
    # cv2.rectangle(image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
    return min_x,max_x,min_y,max_y

def motion_threshold(MOTION_THRESHOLD):
    motion_magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
    if np.max(motion_magnitude) > MOTION_THRESHOLD:
        motion_detected = True
        return motion_detected 
    else:
        motion_detected = False
        return motion_detected 
    
motion_detected = False
frame_counter = 0
photo_name=0
MOTION_THRESHOLD = 5.0
success, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3) as face_mesh:
  while cap.isOpened():

    success, image = cap.read()
    captured_motion=image.copy()
    
    
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    start = time.time()
    frame_counter+=1
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    motion_detected= motion_threshold(MOTION_THRESHOLD)
    prev_gray = gray

    end = time.time()
    fps = 1 / (end - start)
    print(f"{fps:.2f} FPS")
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
    if motion_detected:
        
        cv2.putText(image, "Motion Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if frame_counter%5==0:
            photo_name+=1  
            cv2.putText(captured_motion, str(datetime.datetime.now().strftime("%Y-%m-%d")), (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) # tarih 
            cv2.putText(captured_motion, str(datetime.datetime.now().strftime("%H:%M:%S")), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) # tarih
#################################################################     
            
            folder_name_captured_motion = os.path.join(MOTİON_DETECT, f"Detect_Motion{photo_name}.jpg")
            cv2.imwrite(folder_name_captured_motion,captured_motion)
#################################################################     
            if results.multi_face_landmarks:

                for face_landmarks in results.multi_face_landmarks:
                    
                    min_x,max_x,min_y,max_y= face_rectangle_coordiantes(captured_motion, face_landmarks)
                    cv2.rectangle(image,(min_x,min_y),(max_x,max_y),(0,0,255),2)
                    print(min_x,max_x,min_y,max_y)
                    detect_face = captured_motion[min_y:max_y,min_x:max_x]
                    detect_face=cv2.cvtColor(detect_face,cv2.COLOR_BGR2GRAY)
                    captured_motion=cv2.cvtColor(captured_motion,cv2.COLOR_BGR2GRAY)
#################################################################     
                    folder_name_detect_faces = os.path.join(DETECET_FACES, f"Detect_Face{photo_name}.jpg")
                    cv2.imwrite(folder_name_detect_faces,detect_face)
#################################################################     
                    height, width = captured_motion.shape
                    detect_face= cv2.resize(detect_face, (width, height))
                    merged_image = np.hstack((captured_motion,detect_face))
#################################################################     

                    folder_name_merged = os.path.join(MERGED, f"Merged_Moiton_and_Face{photo_name}.jpg")
                    cv2.imwrite(folder_name_merged,merged_image)
                    cv2.putText(image, "Detected Face Is Recording", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2) # tarih

#################################################################     

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', image)
    cv2.imshow("Motion Track",draw_hsv(flow))
    cv2.imshow("flow",draw_flow(gray,flow))

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
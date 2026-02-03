import cv2
from ultralytics import YOLO
import cvzone
import math

# 1. Initialize the Webcam
cap = cv2.VideoCapture(0)  
cap.set(3, 1280) # Width
cap.set(4, 720)  # Height

# 2. Load the YOLO Model
model = YOLO('yolov8l.pt') 

# 3. Define ALL 80 Class Names
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Threshold for "Overcrowding" (Only applies to 'person')
CROWD_LIMIT = 5 

while True:
    success, img = cap.read()
    if not success:
        break
    
    img = cv2.resize(img, (1280, 720))
        
    # Run detection on all objects
    results = model(img, stream=True, imgsz=1280) 
    
    person_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Class detection
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Confidence check
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf > 0.3:
                # Bounding Box Coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # LOGIC: If it is a person, count them and color code the box
                if currentClass == "person":
                    person_count += 1
                    # Green = Safe, Red = Crowded
                    color = (0, 255, 0) if person_count <= CROWD_LIMIT else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=color)

                # LOGIC: If it is any other object, just detect and label it in blue
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) # Blue box for non-humans
                    cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(255,0,0))

    # --- STATUS DISPLAY ---
    # Display the live PERSON count
    cvzone.putTextRect(img, f'People Count: {person_count}', (50, 50), scale=2, thickness=2, offset=10)

    # Logic: Alert System for People Only
    if person_count > CROWD_LIMIT:
        cvzone.putTextRect(img, 'OVERCROWDING DETECTED', (50, 150), scale=3, thickness=3, colorR=(0, 0, 255), offset=10)

    cv2.imshow("Smart Crowd Monitor", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
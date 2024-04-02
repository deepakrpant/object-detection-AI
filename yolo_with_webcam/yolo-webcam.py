from ultralytics import YOLO
import cv2
import cvzone

model = YOLO('../yolo-weights/yolov8n.pt')
cap = cv2.VideoCapture(0) # 0 is the id of the webcam
cap.set(3, 1280) # width
cap.set(4, 720) # height
while True:
    success, img = cap.read()
    # run the model
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y2), int(x2), int(y2)
            print(x1, y1, x2, y2)
            cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255), thickness = 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
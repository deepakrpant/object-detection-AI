from ultralytics import YOLO
import cv2

# Download weights "n stands for nano". No need to donwlod all the time. others are large and medium
model = YOLO('../yolo-weights/yolov8n.pt')
# model = YOLO('yolov8m.pt')

# run the model
results = model("Images/dp5.jpg", show=True)
cv2.waitKey(0)

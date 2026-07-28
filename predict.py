from ultralytics import YOLO

model = YOLO("runs/detect/train-4/weights/best.pt")

model.predict(
    source="รูปไก่.jpg",   # เปลี่ยนเป็นรูปที่ต้องการทดสอบ
    conf=0.1,
    save=True
)
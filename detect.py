from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/train/weights/best.pt')

# Define path to the image file
source = 'testdata'

# Run inference on the source
results = model(source, save=True)  # list of Results objects
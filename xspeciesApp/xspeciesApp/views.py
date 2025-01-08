import os
import shutil
from PIL import Image
from django.http import StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import cv2
from ultralytics import YOLO
from xspecies.models import RecentScan, FeaturedCategory
import time  # For timing and sleep optimization

# Load YOLOv8 models
leaf_model = YOLO(r"C:\django-tutorial\xspeciesApp\xspeciesApp\leaf.pt", task='detect')
flower_model = YOLO(r"C:\django-tutorial\xspeciesApp\xspeciesApp\flower.pt", task='detect')
fruit_model = YOLO(r"C:\django-tutorial\xspeciesApp\xspeciesApp\fruit.pt", task='detect')

# Global variable to manage the camera feed
camera_active = True

# Function to draw bounding boxes
def draw_bounding_boxes(frame, results, color=(0, 255, 0)):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            label = f"{result.names[class_id]} {confidence:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# Generator for video streaming
def video_stream():
    global camera_active
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        yield b''
        return

    try:
        frame_skip = 2
        frame_count = 0

        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            leaf_results = leaf_model.predict(frame, stream=True)
            flower_results = flower_model.predict(frame, stream=True)
            fruit_results = fruit_model.predict(frame, stream=True)

            frame = draw_bounding_boxes(frame, leaf_results, color=(0, 255, 0))
            frame = draw_bounding_boxes(frame, flower_results, color=(255, 0, 0))
            frame = draw_bounding_boxes(frame, fruit_results, color=(0, 0, 255))

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_data = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')

            time.sleep(0.03)
    finally:
        cap.release()

# View to handle video streaming
def video_feed(request):
    global camera_active
    camera_active = True
    return StreamingHttpResponse(video_stream(), content_type='multipart/x-mixed-replace; boundary=frame')

# View to stop the camera feed
def stop_feed(request):
    global camera_active
    if request.method == "POST":
        camera_active = False
        return JsonResponse({"message": "Camera feed stopped successfully."})
    return JsonResponse({"error": "Invalid request method."})

def save_processed_image(image, filename):
    # Ensure the 'processed' folder exists
    processed_folder = os.path.join(settings.MEDIA_ROOT, 'processed')
    os.makedirs(processed_folder, exist_ok=True)  # Create the directory if it doesn't exist

    # Construct the full path where the image will be saved
    file_path = os.path.join(processed_folder, filename)

    # Convert the OpenCV image (numpy array) to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Save the image to the file system
    pil_image.save(file_path)

    return file_path

def clear_processed_folder():
    processed_folder = os.path.join(settings.MEDIA_ROOT, 'processed')
    # Check if the folder exists
    if os.path.exists(processed_folder):
        # Loop through all files in the folder and delete them
        for filename in os.listdir(processed_folder):
            file_path = os.path.join(processed_folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Error while deleting file {file_path}: {str(e)}")
# View to process gallery image prediction
@csrf_exempt
def predict_image(request):
    if request.method == "POST" and request.FILES.get("image"):
        image_file = request.FILES["image"]
        temp_path = default_storage.save("temp/" + image_file.name, ContentFile(image_file.read()))
        temp_full_path = os.path.join(settings.MEDIA_ROOT, temp_path)

        try:
            image = cv2.imread(temp_full_path)

            # Run predictions
            leaf_results = leaf_model.predict(image)
            flower_results = flower_model.predict(image)
            fruit_results = fruit_model.predict(image)

            # Draw bounding boxes
            for results, color in zip([leaf_results, flower_results, fruit_results], [(0, 255, 0), (255, 0, 0), (0, 0, 255)]):
                image = draw_bounding_boxes(image, results, color=color)

            # Save the processed image in MEDIA_ROOT
           # Save the processed image using the new save_processed_image function
            processed_filename = os.path.basename(temp_path)
            processed_path = save_processed_image(image, processed_filename)
            clear_processed_folder()
            # Return the URL for the processed image
            return JsonResponse({"image_url": settings.MEDIA_URL + 'processed/' + processed_filename})

        except Exception as e:
            return JsonResponse({"error": f"Failed to process image: {str(e)}"})

    return JsonResponse({"error": "Invalid request method."})

# View for the main page
def index(request):
    recent_scans = RecentScan.objects.all()
    categories = FeaturedCategory.objects.all()
    return render(request, 'index.html', {
        'recent_scans': recent_scans,
        'categories': categories,
    })

from flask import Flask, render_template, request, Response, stream_with_context
import cv2
from ultralytics import YOLO
import numpy as np
import math
from flask_socketio import SocketIO, emit


app = Flask(__name__)
socketio = SocketIO(app)

# Load model YOLOv8
model = YOLO("yolo/best.pt")
# Mengatur nilai threshold
model.conf = 0.25  # Confidence threshold
model.iou = 0.45  # Non-maximum suppression (NMS) threshold

# Fungsi untuk mendeteksi objek pada gambar
def detect_objects(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results_list = model(image)
    num_objects = sum(len(result.boxes) for result in results_list)

    image_with_boxes = image.copy()
    for results in results_list:
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.tolist())
                label = "Mangga"
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

    _, encoded_image = cv2.imencode('.jpg', image_with_boxes)
    return encoded_image.tobytes(), num_objects

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/team.html', methods=['GET'])
def team():
    return render_template('team.html')

@app.route('/about.html', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact.html', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/index.html', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/portfolio.html', methods=['GET'])
def portfolio():
    return render_template('portfolio.html')

@app.route('/services.html', methods=['GET'])
def services():
    return render_template('services.html')

@app.route('/result.html', methods=['GET'])
def result():
    return render_template('result.html')

@app.route('/upload.html', methods=['GET'])
def upload_page():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    file_bytes = file.read()

    if file.content_type.startswith('image/'):
        detected_image, total_objects = detect_objects(file_bytes)
        response = Response(detected_image, mimetype='image/jpeg')
        response.headers['Total-Objects'] = str(total_objects)
        return response
    elif file.content_type.startswith('video/'):
        return Response(stream_with_context(generate_video_frames(file_bytes)), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response("Tipe file tidak didukung", status=400)

@app.route('/live-camera.html', methods=['GET'])
def livecamera():
    return render_template('live-camera.html')

@app.route('/video_feed')
def video_feed():
    try:
        return Response(process_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error: {e}")
        return Response("Terjadi kesalahan saat mengakses kamera.", status=500)

active_clients = 0

@socketio.on('connect')
def handle_connect():
    global active_clients
    active_clients += 1
    print(f'Client connected. Total active clients: {active_clients}')

def process_frames():
    cap = cv2.VideoCapture(0)  # Inisialisasi objek kamera di luar loop
    model = YOLO("yolo/best.pt")  # Load YOLOv8 model

    while True:
        ret, frame = cap.read()
        if not ret:
            # Jika tidak dapat membaca frame, lepaskan objek kamera dan buat objek baru
            cap.release()
            cap = cv2.VideoCapture(0)
            continue

        # Detect objects in the frame using YOLOv8
        results = model(frame)

        # Draw bounding boxes around detected objects
        for result in results:
            boxes = result.boxes.xyxy
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                label = "Mangga"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Count the number of detected mangoes
        num_mangoes = sum(len(result.boxes) for result in results)

        # Encode the frame as JPEG and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Kirim jumlah mangga melalui WebSocket
        socketio.emit('object_count', {'num_mangoes': num_mangoes})

    # Lepaskan objek kamera setelah loop selesai
    cap.release()

@socketio.on('disconnect')
def handle_disconnect():
    global active_clients
    active_clients -= 1
    print(f'Client disconnected. Total active clients: {active_clients}')


if __name__ == '__main__':
    app.run(debug=True)
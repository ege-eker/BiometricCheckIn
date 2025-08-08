from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import time
import numpy as np
import os
from facenet_pytorch import MTCNN
from utils import image_to_base64
from client import send_face, register_new_person, add_embedding_to_person_by_id
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file, not implemented yet

app = Flask(__name__)

# Global variables
mtcnn = MTCNN(keep_all=True, device='cpu')
camera = None
camera_active = False
current_frame = None
processed_frame = None
last_faces = []
registration_active = False
person_id = None
registration_data = {}
registration_count = 0

# For thread safety
frame_lock = threading.Lock()


def camera_thread():
    """Thread for capturing video from the camera"""
    global camera, camera_active, current_frame, processed_frame, last_faces

    camera = cv2.VideoCapture(0)
    while camera_active:
        ret, frame = camera.read()
        if not ret:
            print("Couldn't open camera.")
            break

        with frame_lock:
            current_frame = frame.copy()

            # Detecting faces
            faces = detect_faces(current_frame)
            last_faces = faces

            # Prepare the frame for display with detected faces
            display_frame = current_frame.copy()
            for face in faces:
                x, y, w, h = face["box"]
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add text to the frame
            if len(faces) > 0:
                cv2.putText(display_frame, "FACE FOUND", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "Waiting a face...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

            # Registeration mode
            if registration_active:
                cv2.putText(display_frame, f"Register mode: {registration_count + 1}/5 poses", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)

            processed_frame = display_frame

        # wait for a short period to control frame rate
        time.sleep(0.03)  # ~30 FPS

    if camera:
        camera.release()


def detect_faces(frame, threshold=0.95):
    """Detect faces in the given frame using MTCNN."""
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs = mtcnn.detect(rgb_frame)
        faces = []
        if boxes is not None:
            for box, prob in zip(boxes, probs):
                if prob is not None and prob > threshold:
                    x1, y1, x2, y2 = [int(b) for b in box]
                    w, h = x2 - x1, y2 - y1
                    faces.append({"box": (x1, y1, w, h), "prob": prob})
        return faces
    except Exception as e:
        print(f"Face detection error: {str(e)}")
        return []


def generate_frames():
    """Generator function to yield frames for video streaming."""
    global processed_frame

    while camera_active:
        if processed_frame is not None:
            # Make JPG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()

            # Return HTTP multipart response format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame is available, yield a black frame
            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', black_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Refresh rate control
        time.sleep(0.04)  # ~25 FPS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global camera_active

    if not camera_active:
        camera_active = True
        threading.Thread(target=camera_thread, daemon=True).start()
        return jsonify({"success": True, "message": "Cam started"})

    return jsonify({"success": True, "message": "Cam already running"})


@app.route('/recognize_face', methods=['POST'])
def recognize_face():
    global current_frame, last_faces

    with frame_lock:
        if current_frame is None:
            return jsonify({"success": False, "message": "Cam not ready"})

        if not last_faces:
            return jsonify({"success": False, "message": "No face detected in the image"})

        try:
            # Transform the current frame to base64
            base64_img = image_to_base64(current_frame)

            # Send to backend
            recognition_result = send_face(base64_img)

            if recognition_result:
                # Returns the recognition result with additional face location
                recognition_result["face_location"] = last_faces[0]["box"]
                return jsonify({
                    "success": True,
                    "result": recognition_result
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "No face recognized or error in recognition"
                })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error: {str(e)}"
            })



@app.route('/start_registration', methods=['POST'])
def start_registration():
    global registration_active, registration_data, registration_count, person_id

    # Get registration data from request
    data = request.json

    if not data or not all(key in data for key in ["name", "surname", "age", "nationality", "passport_no"]):
        return jsonify({
            "success": False,
            "message": "Missing values in registration data"
        })

    # Check if registration is already active
    registration_active = True
    registration_data = data
    registration_count = 0
    person_id = None

    return jsonify({
        "success": True,
        "message": "Registration started, please capture 5 photos",
    })


@app.route('/capture_registration', methods=['POST'])
def capture_registration():
    """Captures a registration photo and processes it for embedding"""
    global current_frame, registration_active, registration_data, registration_count, person_id, last_faces

    if not registration_active:
        return jsonify({
            "success": False,
            "message": "Registration is not active, please start registration first"
        })

    with frame_lock:
        if current_frame is None:
            return jsonify({
                "success": False,
                "message": "Camera not ready"
            })

        if not last_faces:
            return jsonify({
                "success": False,
                "message": "No face detected in the image"
            })

        try:
            # Transform the current frame to base64
            base64_img = image_to_base64(current_frame)

            if registration_count == 0:
                # first photo - register new person
                response = register_new_person(
                    base64_img,
                    registration_data["name"],
                    registration_data["surname"],
                    int(registration_data["age"]),
                    registration_data["nationality"],
                    registration_data.get("flight_no", ""),
                    registration_data["passport_no"]
                )

                if response.success:
                    person_id = response.person_id
                    registration_count += 1

                    return jsonify({
                        "success": True,
                        "message": f"First photo saved. Person ID: {person_id}",
                        "count": registration_count,
                        "total": 5
                    })
                else:
                    registration_active = False
                    return jsonify({
                        "success": False,
                        "message": f"Register error: {response.message}"
                    })
            else:
                # Other photos - add embedding to existing person
                if person_id is None:
                    registration_active = False
                    return jsonify({
                        "success": False,
                        "message": "Person ID not found, please start registration again"
                    })

                response = add_embedding_to_person_by_id(base64_img, person_id)

                if response.success:
                    registration_count += 1

                    # Check if we have reached the required number of photos
                    if registration_count >= 5:
                        registration_active = False
                        result = {
                            "success": True,
                            "message": "Register completed successfully, 5 photos saved.",
                            "count": registration_count,
                            "total": 5,
                            "completed": True
                        }
                    else:
                        result = {
                            "success": True,
                            "message": f"Pose {registration_count}/5 saved.",
                            "count": registration_count,
                            "total": 5,
                            "completed": False
                        }

                    return jsonify(result)
                else:
                    return jsonify({
                        "success": False,
                        "message": f"Embedding error: {response.message}"
                    })
        except Exception as e:
            return jsonify({
                "success": False,
                "message": f"Error: {str(e)}"
            })


@app.route('/cancel_registration', methods=['POST'])
def cancel_registration():
    global registration_active

    registration_active = False
    return jsonify({
        "success": True,
        "message": "Registration cancelled."
    })


@app.route('/submit_complete_registration', methods=['POST'])
def submit_complete_registration():
    data = request.json

    if not data or not all(key in data for key in ["name", "surname", "age", "nationality", "passport_no", "images"]):
        return jsonify({
            "success": False,
            "message": "Missing values in registration data"
        })

    images = data.get('images', [])
    if len(images) < 5:
        return jsonify({
            "success": False,
            "message": f"Not enough photos, 5 required. {len(images)} sent."
        })

    try:
        # Send grpc requests to register the person and add embeddings
        from client import register_new_person, add_embedding_to_person_by_id

        # Register the first image
        first_image = images[0]
        response = register_new_person(
            first_image,
            data["name"],
            data["surname"],
            int(data["age"]),
            data["nationality"],
            data.get("flight_no", ""),
            data["passport_no"]
        )

        if not response.success:
            return jsonify({
                "success": False,
                "message": f"Registeration failed: {response.message}"
            })

        person_id = response.person_id

        # Add embeddings for the remaining images
        for i, image in enumerate(images[1:], 1):
            embedding_response = add_embedding_to_person_by_id(image, person_id)
            if not embedding_response.success:
                # Log the error but continue processing
                print(f"Warning: Error when adding {i + 1}. embedding: {embedding_response.message}")

        return jsonify({
            "success": True,
            "message": "Registration completed successfully",
            "person_id": person_id
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Error: {str(e)}"
        })


if __name__ == "__main__":
    # Start the camera thread
    camera_active = True
    threading.Thread(target=camera_thread, daemon=True).start()

    # Start the Flask web application
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)

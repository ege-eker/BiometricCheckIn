import cv2
from insightface.app import FaceAnalysis
import os
from dotenv import load_dotenv

load_dotenv()
width = int(os.getenv("DET_SIZE_W", 640))
height = int(os.getenv("DET_SIZE_H", 640))
model_name = os.getenv("MODEL_NAME", "buffalo_l")

class EmbeddingModel:
    def __init__(self):
        # If you are working with a CPU, CPUExecutionProvider is fine. Otherwise, you can use CUDAExecutionProvider for GPU and make some other adjustments with libraries.
        self.app = FaceAnalysis(name=model_name, providers=['CPUExecutionProvider'])
        print("FaceAnalysis det_size for detection:", width, height)
        self.app.prepare(ctx_id=-1, det_size=(width, height))  # -1 for CPU, 0 for GPU, det_size is the size of the detection model input

    def get_embedding(self, image):
        faces = self.app.get(image)
        print(f"Detected faces: {len(faces)}")
        if len(faces) == 0:
            print("No faces detected in the image.")
            return None
        # Choosing the face with the highest detection score
        face = max(faces, key=lambda f: getattr(f, 'det_score', 0.0))

        # Safe crop debug
        img_cropped = getattr(face, "crop", None)
        if img_cropped is not None and hasattr(img_cropped, "size") and img_cropped.size != 0:
            cv2.imwrite("debug_face_crop.jpg", img_cropped)
            print("Debug: Cropped face saved.")
        else:
            # Some models may not have a crop method, so we check for bbox
            bbox = getattr(face, "bbox", None)
            if bbox is not None:
                x1, y1, x2, y2 = [int(val) for val in bbox]
                img_bounding = image[y1:y2, x1:x2]
                if img_bounding is not None and hasattr(img_bounding, "size") and img_bounding.size != 0:
                    cv2.imwrite("debug_face_bbox.jpg", img_bounding)
                    print("Debug: Bounding box face saved.")
                else:
                    print("Debug: Bounding box face is empty or invalid.")
            else:
                print("Debug: Neither crop nor bbox available for the detected face.")
        # Getting and returning the embedding
        embedding = face.normed_embedding  # insightface can return normed embeddings directly
        print(f"Embedding shape: {embedding.shape}")
        return embedding
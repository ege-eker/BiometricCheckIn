import cv2
import base64


def image_to_base64(image):
    """Convert an image to a base64 string."""
    _, buffer = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

import cv2
import json
import logging
import numpy as np
import mediapipe as mp

# Configure logging
logging.basicConfig(level=logging.INFO)

# Face mesh configuration
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Predefined colors for major acupoints
ACUPOINT_COLORS = {
    "YinTang": (255, 0, 0),  # Red
    "SiBai": (0, 255, 0),    # Green
    "ChengJiang": (0, 0, 255),  # Blue
    "FengChi": (255, 0, 255),  # Magenta
}

class AcupointGenerator:
    def __init__(self, data_file):
        try:
            with open(data_file, 'r') as file:
                self.data = json.load(file)
            logging.info(f"Acupoint data successfully loaded from {data_file}")
        except FileNotFoundError as e:
            logging.error(f"File not found: {data_file}. Error: {e}")
            raise
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON from {data_file}. Error: {e}")
            raise
        except Exception as e:
            logging.error(f"Unknown error loading acupoint data from {data_file}. Error: {e}")
            raise

    def generate_acupoints(self, landmarks):
        acupoints = []
        try:
            for acupoint in self.data.get("acupoints", []):
                ref_name = acupoint.get("reference")
                if ref_name in landmarks:
                    ref_point = landmarks[ref_name]
                    offset = acupoint.get("offset", [0, 0])
                    adjusted_point = (int(ref_point[0] + offset[0]), int(ref_point[1] + offset[1]))
                    acupoints.append((acupoint["name"], adjusted_point))
                else:
                    logging.warning(f"Reference point {ref_name} not found in the provided landmarks.")
        except Exception as e:
            logging.error(f"Error generating acupoints: {e}")
            raise
        return acupoints

    def draw_acupoints(self, image, acupoints):
        """
        Draw the acupoints on an image with different colors for each acupoint.
        Args:
            image (numpy array): The image on which to draw the acupoints.
            acupoints (list): List of tuples where each tuple contains the acupoint name and its coordinates (x, y).
        """
        try:
            # Create a transparent overlay for blending
            overlay = image.copy()

            for acupoint in acupoints:
                name, coordinates = acupoint

                # Ensure coordinates are valid tuples with two integer values
                if isinstance(coordinates, tuple) and len(coordinates) == 2:
                    x, y = coordinates
                    if isinstance(x, int) and isinstance(y, int):
                        # Use the predefined color for each acupoint
                        color = ACUPOINT_COLORS.get(name, (200, 200, 200))  # Default light gray

                        # Draw the acupoints as small dots or squares
                        cv2.circle(overlay, (x, y), 3, color, -1, lineType=cv2.LINE_AA)  # Smooth circle

                        # Draw the major acupoint name next to the dot for only major acupoints
                        if name in ACUPOINT_COLORS:
                            cv2.putText(overlay, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                        logging.info(f"Acupoint {name} drawn at ({x}, {y}) with color {color}")

            # Blend the overlay with the original image
            cv2.addWeighted(overlay, 0.4, image, 0.6, 0, image)
        except Exception as e:
            logging.error(f"Error drawing acupoints on image: {e}")
            raise

def apply_face_filter(image):
    """
    Apply smoothing and enhancement filters to make the face more appealing.
    Args:
        image (numpy array): The original image from the webcam.
    Returns:
        numpy array: The image with the filter applied.
    """
    try:
        # Apply a bilateral filter for smoothing while keeping edges sharp
        smooth = cv2.bilateralFilter(image, 15, 75, 75)

        # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Optional: Apply a slight Gaussian blur to give a soft-focus effect
        blurred = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
        
        return blurred
    except Exception as e:
        logging.error(f"Error applying face filter: {e}")
        raise

def get_real_landmarks(frame):
    """
    Detect real facial landmarks using MediaPipe Face Mesh.
    """
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            logging.warning("No face landmarks detected.")
            return {}

        landmarks = {}
        for id, lm in enumerate(results.multi_face_landmarks[0].landmark):
            h, w, _ = frame.shape
            x, y = int(lm.x * w), int(lm.y * h)
            landmarks[f'id_{id}'] = (x, y)
        return landmarks
    except Exception as e:
        logging.error(f"Error during landmark detection: {e}")
        raise

def process_webcam(acupoint_generator):
    """
    Capture frames from the webcam, detect landmarks, apply the filter, and visualize acupoints in real-time.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame from the webcam.")
            break

        try:
            # Apply face filter
            filtered_frame = apply_face_filter(frame)

            # Detect real facial landmarks
            detected_landmarks = get_real_landmarks(filtered_frame)

            # Generate acupoints based on detected landmarks
            acupoints = acupoint_generator.generate_acupoints(detected_landmarks)

            # Draw acupoints on the frame
            acupoint_generator.draw_acupoints(filtered_frame, acupoints)

            # Display the frame with the acupoints and face filter applied
            cv2.imshow('Face Filter & Acupoints Visualization', filtered_frame)

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            break

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize the acupoint generator with the path to the data.json file
        acupoint_generator = AcupointGenerator('data.json')

        # Start the webcam process to visualize acupoints with the face filter in real-time
        process_webcam(acupoint_generator)
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}", exc_info=True)

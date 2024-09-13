import cv2
import logging
from AcupointGenerator import AcupointGenerator, get_real_landmarks
from utils import preprocess_image, run_inference

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_webcam(acupoint_generator):
    """
    Capture frames from the webcam, detect landmarks, and visualize acupoints in real-time.
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
            # Detect real facial landmarks
            detected_landmarks = get_real_landmarks(frame)

            if not detected_landmarks:
                logging.warning("No landmarks detected.")
                continue

            # Generate acupoints based on detected landmarks
            acupoints = acupoint_generator.generate_acupoints(detected_landmarks)

            # Draw acupoints on the frame
            acupoint_generator.draw_acupoints(frame, acupoints)

            # Display the frame with the acupoints
            cv2.imshow('Acupoints Visualization', frame)

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            break

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        # Initialize the acupoint generator with the path to the data.json file
        acupoint_generator = AcupointGenerator('data.json')

        # Start the webcam process to visualize acupoints in real-time
        process_webcam(acupoint_generator)
    except Exception as e:
        logging.critical(f"Critical error in main execution: {e}", exc_info=True)

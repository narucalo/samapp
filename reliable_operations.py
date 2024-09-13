import os
import sys
import cv2
import logging
import json
import time
import numpy as np
import mediapipe as mp
from utils import load_onnx_model_with_tensorrt, preprocess_image, run_inference
from AcupointGenerator import AcupointsGenerator

# Add the current directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

def retry_with_backoff(func, max_retries=5, backoff_factor=1, *args, **kwargs):
    """
    Retry function execution with exponential backoff.
    
    :param func: Function to execute
    :param max_retries: Maximum number of retries
    :param backoff_factor: Base backoff time (seconds)
    :param args: Function arguments
    :param kwargs: Function keyword arguments
    :return: The function result, or raise an exception after retrying
    """
    retries = 0
    while retries < max_retries:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            retries += 1
            delay = backoff_factor * (2 ** retries)
            logging.warning(f"Attempt {retries} failed for {func.__name__}. Retrying in {delay} seconds.")
            time.sleep(delay)
    raise Exception(f"{func.__name__} failed after {max_retries} retries")


def load_model_with_retry(model_path):
    """
    Load ONNX model with retries using exponential backoff.
    """
    return retry_with_backoff(load_onnx_model_with_tensorrt, model_path=model_path)


def initialize_webcam_with_retry():
    """
    Initialize webcam with retries using exponential backoff.
    """
    def init_webcam():
        cap = cv2.VideoCapture(0)  # Using the first webcam
        if not cap.isOpened():
            raise Exception("Webcam initialization failed")
        return cap
    
    return retry_with_backoff(init_webcam)


def main():
    # Load the model with retry
    model_path = 'path_to_your_model.onnx'
    try:
        model = load_model_with_retry(model_path)
        logging.info(f"Model loaded successfully: {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model after retries: {e}")
        return

    # Initialize webcam with retry
    try:
        webcam = initialize_webcam_with_retry()
        logging.info("Webcam initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize webcam after retries: {e}")
        return

    # Initialize Acupoint Generator
    data_file = 'data.json'
    try:
        acupoints_generator = AcupointsGenerator(data_file)
        logging.info("Acupoints generator initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize Acupoints Generator: {e}")
        return

    # Main loop for real-time webcam processing
    while True:
        ret, frame = webcam.read()
        if not ret:
            logging.error("Failed to capture frame from webcam")
            break

        # Preprocess frame and run inference
        preprocessed_frame = preprocess_image(frame)
        landmarks = run_inference(model, preprocessed_frame)

        # Generate acupoints and visualize on the frame
        try:
            acupoints_generator.generate_acupoints(landmarks)
        except Exception as e:
            logging.error(f"Failed to generate acupoints: {e}")

        # Display the frame with acupoints
        cv2.imshow('Acupoint Visualization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    webcam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

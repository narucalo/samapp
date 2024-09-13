import cv2
import numpy as np
import logging
import onnxruntime as ort

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_onnx_model_with_tensorrt(model_path, use_tensorrt=False):
    """
    Load an ONNX model using specified providers.
    If TensorRT is available, it will be used. Otherwise, fallback to CPU.
    """
    providers = ['CPUExecutionProvider']
    if use_tensorrt:
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider'] + providers

    try:
        session = ort.InferenceSession(model_path, providers=providers)
        logging.info(f"Loaded ONNX model from {model_path} with providers: {providers}")
        return session
    except Exception as e:
        logging.error(f"Error loading ONNX model: {e}")
        raise RuntimeError(f"Failed to load ONNX model from {model_path}") from e


def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for ONNX inference.
    Handles grayscale conversion, resizing, and normalization.
    """
    try:
        if len(image.shape) == 2:  # Convert grayscale to RGB if needed
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = cv2.resize(image, target_size)
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # Convert to channel-first format
        logging.info(f"Preprocessed image to shape {image.shape}")
        return np.expand_dims(image, axis=0)  # Add batch dimension
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise


def run_inference(session, input_data):
    """
    Run ONNX inference on the input data, converting it to float16 if necessary and transposing the dimensions.
    
    Args:
        session (onnxruntime.InferenceSession): The ONNX runtime session.
        input_data (numpy array): The input data for the model.
    
    Returns:
        numpy array: The output from the ONNX model inference.
    """
    try:
        # Convert input data to float16 if necessary
        if input_data.dtype != np.float16:
            input_data = input_data.astype(np.float16)

        # Transpose input_data to match expected shape (1, 224, 224, 3)
        input_data = np.transpose(input_data, (0, 2, 3, 1))

        # Create input dictionary for the ONNX model
        inputs = {session.get_inputs()[0].name: input_data}
        outputs = session.run(None, inputs)

        logging.info(f"Inference completed. Output shape: {outputs[0].shape}")
        return outputs[0]
    except Exception as e:
        logging.error(f"Error during ONNX inference: {e}")
        raise


def draw_3d_mesh(image, landmarks):
    """
    Draw a 3D mesh on the image using detected landmarks.
    
    Args:
        image (numpy array): The image where the 3D mesh will be drawn.
        landmarks (list): The list of landmarks as (x, y) coordinates.
    """
    try:
        for landmark in landmarks:
            x, y = int(landmark[0]), int(landmark[1])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)  # Draw small green circles
        logging.info("3D mesh drawing completed.")
    except Exception as e:
        logging.error(f"Error drawing 3D mesh: {e}")
        raise


def detect_and_handle_landmarks(image_path):
    """
    Detect landmarks from an image and preprocess it for the ONNX model.

    Args:
        image_path (str): Path to the image file.

    Returns:
        numpy array: Preprocessed image ready for ONNX inference.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Image not found at {image_path}")

        logging.info(f"Processing image at {image_path}")
        return preprocess_image(image)
    except Exception as e:
        logging.error(f"Error during landmark detection: {e}")
        raise

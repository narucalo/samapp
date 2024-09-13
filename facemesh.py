import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

DOT_COLOR = (255, 255, 255)  # White for dots

def draw_dots(image, landmarks):
    """
    Draw small dots on the face based on the detected landmarks.
    """
    h, w, _ = image.shape

    # Loop through each landmark and draw small dots
    for landmark in landmarks:
        x, y = int(landmark.x * w), int(landmark.y * h)
        
        # Draw a small dot (radius 1)
        cv2.circle(image, (x, y), 1, DOT_COLOR, -1)  # Radius is 1 for smallest possible dot

    return image

def main():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Reduce webcam resolution to lower processing load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw dots on the face mesh landmarks
                frame_with_dots = draw_dots(frame, face_landmarks.landmark)

                # Display the frame with the dots
                cv2.imshow("Face Mesh with Dots", frame_with_dots)
        else:
            # Display frame without landmarks if not detected
            cv2.imshow("Face Mesh with Dots", frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Acupoint AR

Acupoint AR is an augmented reality application that detects and visualizes acupuncture points on the face in real-time. It uses advanced facial detection, alignment, and segmentation to accurately locate acupoints, leveraging the B-cun proportional measurement method for precision.

## Features

- Real-time facial landmark detection using ARKit (iOS) and ARCore (Android).
- Facial segmentation and acupoint visualization.
- 3D rendering of acupoints using Three.js.
- Supports both iOS and Android platforms.
- Camera-based pressure detection on facial regions.

## Technology Stack

- **Framework:** React Native
- **AR (iOS):** ARKit
- **AR (Android):** ARCore
- **Face Tracking:** TensorFlow Lite / OpenCV
- **Camera Access:** React Native Camera
- **3D Rendering:** Three.js
- **User Interaction:** React Native Gesture Handler

## Installation

### Clone the repository:

```bash
git clone https://github.com/narucalo/acupoints.git

npm install

Run on iOS:
npx react-native run-ios

Run on Android:
npx react-native run-android

Usage
Open the app and allow camera access.
The app will detect your face and overlay acupoints on the screen.
You can interact with the points to get more information or adjust the display.
Project Structure
src/components: Contains reusable UI components.
src/screens: Contains different screens of the app.
src/assets: Stores image assets, icons, and other static files.
src/styles: Centralized location for styling (using TailwindCSS).
src/utils: Helper functions and utilities.


This template provides a good starting point. You can add more details specific to your project as needed!

pip install -r requirements.txt

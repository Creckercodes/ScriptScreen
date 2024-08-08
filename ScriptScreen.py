import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Define the green background color
GREEN = (0, 255, 0)

# Open the video capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Retrieve the width and height of the video capture
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a virtual camera
with pyvirtualcam.Camera(FRAME_WIDTH, FRAME_HEIGHT, fps=30) as virtual_camera:
    print('Virtual camera is active.')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Flip the frame horizontally to correct the mirroring
        frame = cv2.flip(frame, 1)

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Perform segmentation
        results = selfie_segmentation.process(rgb_frame)

        if results.segmentation_mask is not None:
            # Create the mask
            mask = results.segmentation_mask > 0.4  # Adjust threshold if needed

            # Convert mask to 3 channels
            mask_3ch = np.stack((mask,) * 3, axis=-1).astype(np.uint8) * 255

            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask_3ch = cv2.morphologyEx(mask_3ch, cv2.MORPH_CLOSE, kernel)
            mask_3ch = cv2.morphologyEx(mask_3ch, cv2.MORPH_OPEN, kernel)

            # Apply Gaussian blur to the mask to smooth edges
            blurred_mask = cv2.GaussianBlur(mask_3ch, (15, 15), 0)
            blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)[1]

            # Create a green background
            green_background = np.zeros_like(small_frame, dtype=np.uint8)
            green_background[:] = GREEN

            # Combine the original frame and the green background using the mask
            output_frame = np.where(blurred_mask == 255, small_frame, green_background)

            # Convert the frame to RGB for pyvirtualcam
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            # Send the frame to the virtual camera
            virtual_camera.send(output_frame_rgb)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture
cap.release()
cv2.destroyAllWindows()

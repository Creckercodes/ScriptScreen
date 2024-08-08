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

# Frame size and processing parameters
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

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
            mask = results.segmentation_mask

            # Apply a lower threshold to the mask to include more parts of the body
            condition = mask > 0.3

            # Apply morphological operations to clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            condition = cv2.morphologyEx(condition.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            condition = cv2.morphologyEx(condition, cv2.MORPH_OPEN, kernel)

            # Refine the mask using erosion and dilation
            eroded_mask = cv2.erode(condition, kernel, iterations=1)
            expanded_mask = cv2.dilate(eroded_mask, kernel, iterations=1)
            
            # Use contours to refine mask boundaries
            contours, _ = cv2.findContours(expanded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                refined_mask = np.zeros_like(expanded_mask)
                cv2.drawContours(refined_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            else:
                refined_mask = expanded_mask

            # Smooth the mask using Gaussian blur
            blurred_mask = cv2.GaussianBlur(refined_mask, (15, 15), 0)
            blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)[1]

            # Create a green background
            green_background = np.zeros_like(small_frame, dtype=np.uint8)
            green_background[:] = GREEN

            # Combine the frame with the green background using the smoothed mask
            output_frame = np.where(blurred_mask[:, :, None], small_frame, green_background)

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

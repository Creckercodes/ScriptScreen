import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from tkinter import Tk, filedialog, Button, Label, Frame

# Function to select file
def select_file(file_type):
    if file_type == 'image':
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    elif file_type == 'video':
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
    else:
        file_path = None
    return file_path

def set_background_to_image():
    global background_mode, background_img, background_video
    background_path = select_file('image')
    if background_path:
        background_img = cv2.imread(background_path)
        background_img = cv2.resize(background_img, (FRAME_WIDTH, FRAME_HEIGHT))
        background_video = None
        background_mode = 'image'

def set_background_to_video():
    global background_mode, background_img, background_video
    background_path = select_file('video')
    if background_path:
        background_video = cv2.VideoCapture(background_path)
        background_img = None
        background_mode = 'video'

def set_background_to_green():
    global background_mode, background_img, background_video
    background_img = None
    background_video = None
    background_mode = 'green'

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Open the video capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Retrieve the width and height of the video capture
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the green background color
GREEN = (0, 255, 0)

# Initial background mode is green screen
background_mode = 'green'
background_img = None
background_video = None

# Create the Tkinter window
root = Tk()
root.title("Background Selector")

# Create the GUI components
frame = Frame(root)
frame.pack(padx=10, pady=10)

Label(frame, text="Select Background:").grid(row=0, columnspan=2)

Button(frame, text="Green Screen", command=set_background_to_green).grid(row=1, column=0, padx=5, pady=5)
Button(frame, text="Image", command=set_background_to_image).grid(row=1, column=1, padx=5, pady=5)
Button(frame, text="Video", command=set_background_to_video).grid(row=2, column=0, columnspan=2, padx=5, pady=5)

# Create a virtual camera
with pyvirtualcam.Camera(FRAME_WIDTH, FRAME_HEIGHT, fps=30) as virtual_camera:
    print('Virtual camera is active.')

    def update_camera():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            root.after(1, update_camera)
            return

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
            mask = results.segmentation_mask > 0.45  # Adjust threshold if needed

            # Convert mask to 3 channels
            mask_3ch = np.stack((mask,) * 3, axis=-1).astype(np.uint8) * 255

            # Clean up the mask
            kernel = np.ones((5, 5), np.uint8)
            mask_3ch = cv2.morphologyEx(mask_3ch, cv2.MORPH_CLOSE, kernel)
            mask_3ch = cv2.morphologyEx(mask_3ch, cv2.MORPH_OPEN, kernel)

            # Apply Gaussian blur to the mask to smooth edges
            blurred_mask = cv2.GaussianBlur(mask_3ch, (15, 15), 0)
            blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)[1]

            # Select background frame
            if background_mode == 'image' and background_img is not None:
                background_frame = background_img
            elif background_mode == 'video' and background_video is not None:
                ret_bg, background_frame = background_video.read()
                if not ret_bg:
                    background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret_bg, background_frame = background_video.read()
                background_frame = cv2.resize(background_frame, (FRAME_WIDTH, FRAME_HEIGHT))
            else:
                background_frame = np.zeros_like(small_frame, dtype=np.uint8)
                background_frame[:] = GREEN

            # Combine the original frame and the background using the mask
            output_frame = np.where(blurred_mask == 255, small_frame, background_frame)

            # Convert the frame to RGB for pyvirtualcam
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            # Send the frame to the virtual camera
            virtual_camera.send(output_frame_rgb)

        root.after(1, update_camera)

    root.after(1, update_camera)
    root.mainloop()

# Release the video capture
cap.release()
cv2.destroyAllWindows()
if background_video is not None:
    background_video.release()

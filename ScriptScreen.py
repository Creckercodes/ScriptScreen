import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
from tkinter import Tk, filedialog, Button, Label, Frame, Scale, HORIZONTAL
import threading
import time
import pygame
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from pydub.playback import play
import threading

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
    global background_mode, background_img, background_video, audio_thread
    background_path = select_file('image')
    if background_path:
        background_img = cv2.imread(background_path)
        background_img = cv2.resize(background_img, (FRAME_WIDTH, FRAME_HEIGHT))
        background_video = None
        background_mode = 'image'
        stop_audio()

def set_background_to_video():
    global background_mode, background_img, background_video, video_fps, audio_clip, audio_thread
    background_path = select_file('video')
    if background_path:
        background_video = cv2.VideoCapture(background_path)
        if not background_video.isOpened():
            print("Error: Could not open video file")
            return

        video_fps = background_video.get(cv2.CAP_PROP_FPS)
        if video_fps == 0:
            video_fps = 30  # Default FPS if video FPS could not be retrieved
        background_img = None
        background_mode = 'video'

        # Extract and play audio
        extract_audio(background_path)
        start_audio()

def set_background_to_green():
    global background_mode, background_img, background_video, audio_thread
    background_img = None
    background_video = None
    background_mode = 'green'
    stop_audio()

def extract_audio(video_path):
    global audio_clip
    video = VideoFileClip(video_path)
    audio = video.audio
    audio_path = video_path.replace('.mp4', '.wav')  # Save as .wav file
    audio.write_audiofile(audio_path)
    audio_clip = audio_path

def start_audio():
    global audio_clip, audio_thread
    if audio_clip:
        def play_audio():
            audio = AudioSegment.from_wav(audio_clip)
            play(audio)
        # Start audio in a separate thread
        audio_thread = threading.Thread(target=play_audio)
        audio_thread.start()

def stop_audio():
    global audio_thread
    if audio_thread and audio_thread.is_alive():
        pygame.mixer.music.stop()
        audio_thread.join()

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Limit buffer size to 1 frame

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
video_fps = 30  # Default video FPS
audio_clip = None
audio_thread = None

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

# Add a slider to adjust the segmentation threshold
threshold_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, label="Segmentation Threshold")
threshold_slider.set(40)  # Set initial value to 40 (corresponds to 0.4)
threshold_slider.pack()

# Create a virtual camera
with pyvirtualcam.Camera(FRAME_WIDTH, FRAME_HEIGHT, fps=30) as virtual_camera:
    print('Virtual camera is active.')

    def update_camera():
        global background_video, video_fps
        start_time = time.time()

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
        seg_results = selfie_segmentation.process(rgb_frame)
        pose_results = pose.process(rgb_frame)

        if seg_results.segmentation_mask is not None:
            # Get the threshold value from the slider
            threshold = threshold_slider.get() / 100.0

            # Create the initial mask
            mask = seg_results.segmentation_mask > threshold
            
            # Convert the mask to uint8 for drawing
            mask_uint8 = (mask * 255).astype(np.uint8)

            # Add pose landmarks to refine the mask
            if pose_results.pose_landmarks:
                for landmark in pose_results.pose_landmarks.landmark:
                    x = int(landmark.x * FRAME_WIDTH)
                    y = int(landmark.y * FRAME_HEIGHT)
                    cv2.circle(mask_uint8, (x, y), 10, 255, -1)

            # Use contours to get a precise mask
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            precise_mask = np.zeros_like(mask_uint8)
            cv2.drawContours(precise_mask, contours, -1, 255, thickness=cv2.FILLED)

            # Convert mask back to 3 channels for processing
            mask_3ch = np.stack((precise_mask,) * 3, axis=-1)

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
                if ret_bg:
                    background_frame = cv2.resize(background_frame, (FRAME_WIDTH, FRAME_HEIGHT))
                else:
                    background_frame = np.zeros_like(small_frame, dtype=np.uint8)
                    background_frame[:] = GREEN
            else:
                background_frame = np.zeros_like(small_frame, dtype=np.uint8)
                background_frame[:] = GREEN

            # Combine the original frame and the background using the mask
            output_frame = np.where(blurred_mask == 255, small_frame, background_frame)

            # Convert the frame to RGB for pyvirtualcam
            output_frame_rgb = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)

            # Send the frame to the virtual camera
            virtual_camera.send(output_frame_rgb)

        # Calculate the time taken and sleep if needed to match the FPS
        elapsed_time = time.time() - start_time
        delay = max(1.0 / video_fps - elapsed_time, 0)
        time.sleep(delay)
        
        root.after(1, update_camera)

    root.after(1, update_camera)
    root.mainloop()

# Release the video capture and background video
cap.release()
cv2.destroyAllWindows()
if background_video is not None:
    background_video.release()

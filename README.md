# ScriptScreen
Realtime green screen around you no more setting up lighting and keying.

Instead of having to get the perfect lighting and set up a green screen this program does it without all that tinkering and setup! First run the script, Then the program will make a virtual camera of you green screened in Realtime. you can open the camera in anything. Open it in chrome, OBS, LIV, and just about whatever. This program works well with fast action as well. Also you can upload your own videos and images to it or just stay in green screen mode.

# Needed Depedencys:

cv2,
mediapipe,
pyvirtualcam,
numpy,

# Disclaimers:
Its not perfect the more objects in the veiw and that are closer to you may appear. This does not mean you need a green screen.


# Test Videos:

https://github.com/user-attachments/assets/e72630a2-fb9e-4801-bc31-61a0539ad914

# Known Issues:
Hand and feet detection issues where the hands and feet arent being added to the mask well. Reason being: Its Mediapipe, Mediapipe had been knowen to have detection issues with hands and feet espeically when they are closer to the outside of the frame.
Im not sure how to reslove this.

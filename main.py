import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
import os
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")

col1, col2 = st.columns([3,2])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
 
with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

genai.configure(api_key="AIzaSyDhTGPMmKhnEIWW-sUsmi9ZzoK_8a8iRDM")
model = genai.GenerativeModel("gemini-1.5-flash")


# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize hand detector
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Function to get hand information
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=False, flipType=True)

    if hands:
        hand = hands[0]  
        lmList = hand["lmList"]  # Landmark list
        fingers = detector.fingersUp(hand)  # Fingers up status
        return fingers, lmList
    return None

# Function to draw on the canvas
def drawOnCanvas(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    
    # Check for specific finger state to start drawing
    if fingers == [0, 1, 0, 0, 0]:  # Index finger up
        current_pos = lmList[8][0:2]  # Index finger's current position (x, y)
        if prev_pos is None:
            prev_pos = current_pos
        # Draw a line between the previous and current positions
        cv2.line(canvas, prev_pos, current_pos, (255, 0, 255), 10)
    elif fingers == [1,1,1,1,1]:
        canvas = np.zeros_like(img)


    return current_pos, canvas

def getAnswerFromAI(model, info, canvas):
    fingers, lmList = info
    if fingers == [1,0,0,0,0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math Problem", pil_image])
        print(response.text)
        return response.text

# Variables
prev_pos = None
canvas = None 
output_text = ""

while True: 
    success, img = cap.read()
    if not success:
        break

    # Initialize canvas
    if canvas is None:
        canvas = np.zeros_like(img)

    # Get hand information
    info = getHandInfo(img)

    if info:
        # Draw on the canvas based on the hand gesture
        prev_pos, canvas = drawOnCanvas(info, prev_pos, canvas)
        output_text = getAnswerFromAI(model=model,info=info, canvas=canvas)

    if output_text:
        output_text_area.text(output_text)  

    # Combine the original image and the drawing canvas
    image_combine = cv2.addWeighted(src1=img, alpha=0.75, src2=canvas, beta=0.25, gamma=0)

    # Display the combined image
    # cv2.imshow("AIGesture Calculator", image_combine)
    FRAME_WINDOW.image(image_combine,channels="BGR")
    # cv2.imshow("Canvas",canvas)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

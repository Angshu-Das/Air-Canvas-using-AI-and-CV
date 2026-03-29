import cv2
import mediapipe as mp
import numpy as np
import time
import os
import math

# -----------------------------
# Setup
# -----------------------------
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

mpHands = mp.solutions.hands
hands = mpHands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

canvas = np.zeros((720,1280,3), np.uint8)

xp, yp = 0,0

color = (0,255,255)
thickness = 10

if not os.path.exists("saved_drawings"):
    os.makedirs("saved_drawings")

# -----------------------------
# Helper Functions
# -----------------------------
def fingers_up(handLms):
    fingers = []

    # Index finger
    if handLms.landmark[8].y < handLms.landmark[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle finger
    if handLms.landmark[12].y < handLms.landmark[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers


def find_distance(x1,y1,x2,y2):
    return math.hypot(x2-x1,y2-y1)

# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = cap.read()
    frame = cv2.flip(frame,1)

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    # -----------------------------
    # Draw Menu
    # -----------------------------
    cv2.rectangle(frame,(0,0),(150,65),(0,255,255),-1)
    cv2.putText(frame,"YELLOW",(10,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    cv2.rectangle(frame,(150,0),(300,65),(0,255,0),-1)
    cv2.putText(frame,"GREEN",(170,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    cv2.rectangle(frame,(300,0),(450,65),(0,0,255),-1)
    cv2.putText(frame,"RED",(330,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.rectangle(frame,(450,0),(600,65),(0,0,0),-1)
    cv2.putText(frame,"ERASER",(470,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.rectangle(frame,(600,0),(750,65),(200,200,200),-1)
    cv2.putText(frame,"CLEAR",(620,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    cv2.rectangle(frame,(750,0),(900,65),(255,200,0),-1)
    cv2.putText(frame,"SAVE",(790,40),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,0),2)

    # -----------------------------
    # Hand Detection
    # -----------------------------
    if results.multi_hand_landmarks:

        for handLms in results.multi_hand_landmarks:

            h, w, c = frame.shape

            cx = int(handLms.landmark[8].x * w)
            cy = int(handLms.landmark[8].y * h)

            thumbx = int(handLms.landmark[4].x * w)
            thumby = int(handLms.landmark[4].y * h)

            cv2.circle(frame,(cx,cy),10,(255,0,255),cv2.FILLED)

            fingers = fingers_up(handLms)

            # -----------------------------
            # PINCH → THICKNESS CONTROL
            # -----------------------------
            distance = find_distance(cx,cy,thumbx,thumby)

            thickness = int(np.interp(distance,[20,200],[5,50]))

            cv2.line(frame,(cx,cy),(thumbx,thumby),(255,0,255),2)

            # -----------------------------
            # SELECTION MODE
            # -----------------------------
            if fingers[0] == 1 and fingers[1] == 1:

                xp, yp = 0,0

                cv2.putText(frame,"SELECTION MODE",(950,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),2)

                if cy < 65:

                    if 0 < cx <150:
                        color = (0,255,255)

                    elif 150 < cx <300:
                        color = (0,255,0)

                    elif 300 < cx <450:
                        color = (0,0,255)

                    elif 450 < cx <600:
                        color = (0,0,0)

                    elif 600 < cx <750:
                        canvas = np.zeros((720,1280,3), np.uint8)

                    elif 750 < cx <900:
                        filename = f"saved_drawings/drawing_{int(time.time())}.png"
                        cv2.imwrite(filename, canvas)
                        print("Saved:", filename)

            # -----------------------------
            # DRAW MODE
            # -----------------------------
            elif fingers[0] == 1 and fingers[1] == 0:

                cv2.putText(frame,"DRAW MODE",(950,40),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

                if xp == 0 and yp == 0:
                    xp, yp = cx, cy

                cv2.line(canvas,(xp,yp),(cx,cy),color,thickness)

                xp, yp = cx, cy

            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)

    # -----------------------------
    # Merge Canvas
    # -----------------------------
    gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray,50,255,cv2.THRESH_BINARY_INV)

    inv = cv2.cvtColor(inv,cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame,inv)
    frame = cv2.bitwise_or(frame,canvas)

    cv2.imshow("AI Air Drawing Board",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
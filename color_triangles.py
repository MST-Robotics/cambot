import cv2
import time
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera

camera = PiCamera()
rawCapture = PiRGBArray(camera)
time.sleep(0.1) # warm up
 

while true:
    camera.capture(rawCapture, format="bgr")
    image_obj = rawCapture.array

    gray = cv2.cvtColor(image_obj, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(gray, kernel, iterations=1)

    blur = cv2.GaussianBlur(dilation, (5, 5), 0)


    thresh = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)

    # Now finding Contours         ###################
    _, contours, _ = cv2.findContours(
        thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    tri_coords = []
    sq_coords = []
    cc_coords = []
    for cnt in contours:
            # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
        approx = cv2.approxPolyDP(
            cnt, 0.07 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3:
            tri_coords.append([cnt])
        elif len(approx) == 4:
            sq_coords.append([cnt])
            
    # color contors
    for cnt in tri_coords:
        cv2.drawContours(image_obj, cnt, 0, (0, 0, 255), 3)
    for cnt in tri_coords:
        cv2.drawContours(image_obj, cnt, 0, (0, 255, 0), 3)
    for cnt in tri_coords:
        cv2.drawContours(image_obj, cnt, 0, (255, 0, 0), 3)

    cv2.imshow("result.png", image_obj)

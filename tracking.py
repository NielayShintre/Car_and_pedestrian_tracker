import cv2
img_file = 'car.jpg'
car_tracker_file = 'cars.xml'
car_tracker = cv2.CascadeClassifier(car_tracker_file)

pedestrian_tracker_file = 'haarcascade_fullbody.xml'
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#img = cv2.imread(img_file)
#video = cv2.VideoCapture('Tesla Autopilot Dashcam.mp4')
video2 = cv2.VideoCapture('Pedestrian video.mp4')
while True:
    (read_successful, frame) = video2.read()
    if read_successful:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(grayscale_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscale_frame)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    for (x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 225, 255), 3)

    cv2.imshow("Detector", frame)
    cv2.waitKey(1)

'''black_n_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Create car classifier

#detect cars


cv2.imshow("Detector", img)
cv2.waitKey()
'''


print("Code Complete")
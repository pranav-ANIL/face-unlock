import cv2
import face_recognition
import os
if not os.path.exists(r"C:\Users\prana\Desktop\face_emotion_detection_pr\my_face.jpg"):
    print("no registered face found")
    exit()
known_image = face_recognition.load_image_file(r"C:\Users\prana\Desktop\face_emotion_detection_pr\my_face.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]
cam = cv2.VideoCapture(0)
print("show your face to unlock")
while True:
    ret,frame = cam.read()
    if not ret:
        continue
    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    fcae_loactions = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb,fcae_loactions)
    for face_encoding in face_encodings:
        match  = face_recognition.compare_faces([known_encoding],face_encoding,tolerance=0.5)
        if match[0]:
            print("face matched.access granted")
            cv2.destroyAllWindows()
            exit()
    cv2.imshow("face unlock",frame)
    if cv2.waitKey(1)==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()     
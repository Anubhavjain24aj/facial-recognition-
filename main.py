import face_recognition
import cv2
import csv
import numpy as np
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#Load known faces
Kunal_image = face_recognition.load_image_file("faces/Kunal.jpg")
Kunal_encoding = face_recognition.face_encodings(Kunal_image)[0]
Anubhav_image = face_recognition.load_image_file("faces/Anubhav.jpg")
Anubhav_encoding = face_recognition.face_encodings(Anubhav_image)[0]
Dhawal_image = face_recognition.load_image_file("faces/Dhawal.jpg")
Dhawal_encoding = face_recognition.face_encodings(Dhawal_image)[0]
Eshaan_image = face_recognition.load_image_file("faces/Eshaan.jpg")
Eshaan_encoding = face_recognition.face_encodings(Eshaan_image)[0]
Shivam_image = face_recognition.load_image_file("faces/Shivam.jpg")
Shivam_encoding = face_recognition.face_encodings(Shivam_image)[0]

know_face_encodings = [Kunal_encoding,Anubhav_encoding,Dhawal_encoding,Eshaan_encoding,Shivam_encoding]
know_face_names = ["Kunal","Anubhav","Dhawal","Eshaan","Shivam"]

# list of expected detection
Expected = know_face_names.copy()

face_locations = []
face_encodings = []

# get current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv" , "w+" , newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    # recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(know_face_encodings,face_encoding)
        face_distance = face_recognition.face_distance(know_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = know_face_names[best_match_index]

        # Add the text if a person is detected
        if name in know_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,100)
            fontScale = 1.5
            fontColor = (235,0,0)
            thickness = 3
            lineType = 2
            cv2.putText (frame,name + " Detected",bottomLeftCornerOfText,font,fontScale,fontColor,thickness,lineType)

            if name in Expected:
                Expected.remove(name)
                current_time = now.strftime("%H-%M%S")
                lnwriter.writerow([name,current_time])

    cv2.imshow("Detected",frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()


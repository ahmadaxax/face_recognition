import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime 
output_directory = "face_recognition\\attendance_records"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)
    
video_capture=cv2.VideoCapture(0)

ahmad_image=face_recognition.load_image_file("faces\\ahmad.jpg")
ahmad_encoding=face_recognition.face_encodings(ahmad_image)[0]

arman_image=face_recognition.load_image_file("faces\\arman.jpg")
arman_encoding=face_recognition.face_encodings(arman_image)[0]

zaid_image=face_recognition.load_image_file("faces\\zaid.jpg")
zaid_encoding=face_recognition.face_encodings(zaid_image)[0]


hammad_image=face_recognition.load_image_file("faces\\hammad.jpg")
hammad_encoding=face_recognition.face_encodings(hammad_image)[0]

hamza_image=face_recognition.load_image_file("faces\\hamza.jpg")
hamza_encoding=face_recognition.face_encodings(hamza_image)[0]

khalid_image=face_recognition.load_image_file("faces\\khalid.jpg")
khalid_encoding=face_recognition.face_encodings(khalid_image)[0]


known_face_encodings=[ahmad_encoding,arman_encoding,zaid_encoding,hammad_encoding,hamza_encoding,khalid_encoding]
known_face_names=["ahmad","arman","zaid","hammad","hamza","khalid"]

students=known_face_names.copy()

face_locations=[]
face_encodings=[]

now=datetime.now()
current_date=now.strftime("%Y-%m-%d")
current_time=now.strftime("%H-%M-%S")

file_path = os.path.join(output_directory, f"{current_date}.csv")


f = open(file_path, "w+", newline="")
lnwriter=csv.writer(f)
while True:
    _,frame=video_capture.read()
    small_frame=cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame=cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)
    
    face_locations=face_recognition.face_locations(rgb_small_frame)
    face_encodings=face_recognition.face_encodings(rgb_small_frame,face_locations)
    
    for face_encoding in face_encodings:
     matches=face_recognition.compare_faces(known_face_encodings,face_encoding)
     face_distance=face_recognition.face_distance(known_face_encodings,face_encoding)
     best_match_index=np.argmin(face_distance)
     
     if(matches[best_match_index]):
         name=known_face_names[best_match_index]
    
         if name in known_face_names:
          font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
          bottomleftcornoroftext=(10,100)
          fontScale=1.5
          fontcolor=(255,0,80)
          thickness=3
          linetype=2
          cv2.putText(frame,name+"present",bottomleftcornoroftext,font,fontScale,fontcolor,thickness,linetype)         
     
         if name in students:
          students.remove(name)
          lnwriter.writerow([name,current_date,current_time])  
          
    cv2.imshow("attendance",frame)
    if cv2.waitKey(1) & 0xFF == ord(" "):
         video_capture.release()
         cv2.destroyAllWindows()  
         break
f.close()   
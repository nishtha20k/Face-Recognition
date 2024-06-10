import cv2
import face_recognition

reference_image = face_recognition.load_image_file("C:/Users/HP/Desktop/1.jpg")
reference_encoding = face_recognition.face_encodings(reference_image)[0]

video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for face_encoding in face_encodings:
        match = face_recognition.compare_faces([reference_encoding], face_encoding)

        if match[0]:
            color = (0, 255, 0)  
        else:
            color = (0, 0, 255)  

        top, right, bottom, left = face_locations[0]
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
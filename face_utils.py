import cv2

face_cascade = cv2.CascadeClassifier(
    "har.xml"
)

def extract_faces(image):
    faces = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(64, 64)
    )

    for (x, y, w, h) in detections:
        face = image[y:y+h, x:x+w]
        if face.size > 0:
            faces.append(face)

    return faces

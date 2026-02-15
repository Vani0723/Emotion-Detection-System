import cv2
from keras.models import load_model
import numpy as np
import os
from datetime import datetime

model = load_model("facialemotionmodel.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Create a folder to save detected faces
save_dir = "emotion_history"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

print("Press 'q' to quit...")

while True:
    ret, im = webcam.read()
    if not ret:
        break

    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (p, q, r, s) in faces:
        face_img = gray[q:q+s, p:p+r]
        cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

        face_img_resized = cv2.resize(face_img, (48, 48))
        img = extract_features(face_img_resized)
        pred = model.predict(img)
        emotion = labels[pred.argmax()]

        # Display emotion on frame
        cv2.putText(im, emotion, (p, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)

        # Save frame with timestamp and emotion
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{emotion}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, im)

    # Show video output
    cv2.imshow("Emotion Detection", im)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()

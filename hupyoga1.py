import cv2
import numpy as np
from tensorflow import keras
from keras.models import load_model
import os



def predict(stop_flag):

    emotion_model = load_model("HUPYogadeep3.h5")  
    emotion_labels = ['downdog', 'goddess','plank','tree','warrior2','wrong-downdog', 'wrong-goddess','wrong-plank','wrong-tree','wrong-warrior2']

    #getting image names from poses folder
    images = []
    for image in os.listdir('poses'):
        name, ext = os.path.splitext(image)
        images.append(name)



    cap = cv2.VideoCapture(0)

    while stop_flag.value:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            break

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Resize grayscale frame to fit your emotion model's input size
        resized_gray = cv2.resize(gray, (60, 60))
        resized_gray = resized_gray / 255.0
        resized_gray = np.expand_dims(resized_gray, axis=0)
        resized_gray = np.expand_dims(resized_gray, axis=-1)  # Add an extra dimension for channel

        # Perform emotion prediction on the resized grayscale frame
        emotion_prediction = emotion_model.predict(resized_gray)
        emotion_index = np.argmax(emotion_prediction)
        emotion = emotion_labels[emotion_index]

        # Display the emotion prediction on the frame
        cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


        if 'wrong' in emotion:
            for entry in images:
                if entry in emotion:
                    small_image = cv2.imread(f'poses/{entry}.jpeg')
                    cv2.putText(frame, f"{entry}", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # Get the dimensions of the small image
                    small_height, small_width, _ = small_image.shape
                    small_height, small_width, _ = small_image.shape
                    small_height = int(small_height * 0.25)  # Reduce height to 50%
                    small_width = int(small_width * 0.10)  # Reduce width to 50%
                    small_image = cv2.resize(small_image, (small_width, small_height))
            x_offset = frame.shape[1] - small_width - 10  
            y_offset = frame.shape[0] - small_height - 10
            frame[y_offset:y_offset+small_height, x_offset:x_offset+small_width] = small_image

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


    cap.release()
    cv2.destroyAllWindows()

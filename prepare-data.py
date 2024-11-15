import os
import cv2
import numpy as np
import time
from utils import get_face_landmarks

# Configuración de rutas y extensiones
data_dir = '/Users/kevinguzmanhuamani/Desktop/computer_vision/Emotion-Recognition/faces'
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
output_file = 'data.txt'

# Procesamiento por lotes con guardado de archivos de emociones
def process_batch(image_paths, emotion_indx, batch_number, emotion_name):
    batch_output = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            continue

        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            batch_output.append(face_landmarks)

    if batch_output:
        batch_filename = f"{emotion_name}_batch_{batch_number}.txt"
        np.savetxt(batch_filename, np.asarray(batch_output))
        print(f"Lote {batch_number} de {emotion_name} guardado como {batch_filename}")

# Procesa cada emoción en dos lotes
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    emotion_dir = os.path.join(data_dir, emotion)
    if os.path.isdir(emotion_dir):
        image_paths = [
            os.path.join(emotion_dir, img)
            for img in os.listdir(emotion_dir)
            if img.lower().endswith(valid_extensions)
        ]

        mid_index = len(image_paths) // 2
        process_batch(image_paths[:mid_index], emotion_indx, 1, emotion)
        time.sleep(5)
        process_batch(image_paths[mid_index:], emotion_indx, 2, emotion)

        # Concatenación de los lotes en un archivo final por emoción
        emotion_filename = f"{emotion}.txt"
        with open(emotion_filename, 'w') as emotion_file:
            for batch_number in range(1, 3):
                batch_filename = f"{emotion}_batch_{batch_number}.txt"
                if os.path.exists(batch_filename):
                    with open(batch_filename, 'r') as batch_file:
                        emotion_file.write(batch_file.read())
                    os.remove(batch_filename)

# Concatenación de archivos de todas las emociones
with open(output_file, 'w') as outfile:
    for emotion in sorted(os.listdir(data_dir)):
        emotion_filename = f"{emotion}.txt"
        if os.path.exists(emotion_filename):
            with open(emotion_filename, 'r') as infile:
                outfile.write(infile.read())
            os.remove(emotion_filename)

import pickle
import cv2
import gradio as gr
from gradio_webrtc import WebRTC
from utils import get_face_landmarks

# Diccionario para mapear los outputs numéricos a emociones
emotion_map = {1: 'angry', 2: 'happy', 3: 'surprised'}

# Cargar el modelo entrenado
with open('./model', 'rb') as f:
    model = pickle.load(f)

def process_landmarks(img, model, emotion_map):
    """Procesa la imagen para detectar landmarks y clasificar la emoción."""
    H, W, _ = img.shape

    # Obtener landmarks faciales
    face_landmarks = get_face_landmarks(img, draw=True, static_image_mode=False)

    # Validar que se obtuvieron los landmarks necesarios
    if len(face_landmarks) == 1404:
        try:
            output = model.predict([face_landmarks])
            emotion_text = emotion_map.get(int(output[0]), "Unknown")
        except Exception as e:
            print("Error en la predicción:", e)
            emotion_text = "Error"
    else:
        emotion_text = "No face detected"

    # Añadir el texto de la emoción en la imagen
    cv2.putText(
        img, emotion_text, (10, H - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4
    )

    return img

# Función principal para Gradio
def detection(image):
    processed_image = process_landmarks(image, model, emotion_map)
    return cv2.resize(processed_image, (1024, 1024))

css = """.my-group {max-width: 600px !important; max-height: 600 !important;}
                      .my-column {display: flex !important; justify-content: center !important; align-items: center !important};"""

with gr.Blocks(css=css) as demo:
    gr.HTML(
        """
    <h1 style='text-align: center'>
    Emotion Recognition Webcam Stream (Powered by WebRTC ⚡️)
    </h1>
    """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        Detect Emotions in Real-Time Using Face Landmarks
        </h3>
        """
    )
    with gr.Column(elem_classes=["my-column"]):
        with gr.Group(elem_classes=["my-group"]):
            image = WebRTC(label="Stream", rtc_configuration=None)

        # Configuración del stream para ejecutar la función de detección
        image.stream(
            fn=detection, inputs=[image], outputs=[image], time_limit=50
        )

if __name__ == "__main__":
    demo.launch(share=True)

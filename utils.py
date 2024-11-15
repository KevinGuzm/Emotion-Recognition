import cv2
import mediapipe as mp


def get_face_landmarks(image, draw=False, static_image_mode=True):
    # Convierte la imagen a RGB
    image_input_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Configura el modelo de face_mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=static_image_mode,
                                                max_num_faces=1,
                                                min_detection_confidence=0.5)

    results = face_mesh.process(image_input_rgb)
    face_mesh.close()  # Cierra el recurso después de procesar la imagen

    image_landmarks = []

    # Si se detectan landmarks faciales
    if results.multi_face_landmarks:
        if draw:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=results.multi_face_landmarks[0],
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec
            )

        ls_single_face = results.multi_face_landmarks[0].landmark
        xs_ = [idx.x for idx in ls_single_face]
        ys_ = [idx.y for idx in ls_single_face]
        zs_ = [idx.z for idx in ls_single_face]

        for j in range(len(xs_)):
            image_landmarks.extend([
                xs_[j] - min(xs_),
                ys_[j] - min(ys_),
                zs_[j] - min(zs_)
            ])

    return image_landmarks

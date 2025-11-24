import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def _compute_eye_centers(landmarks, image_w, image_h):
    """Compute the pixel coordinates of the left and right eye centers.

    Args:
        landmarks (mediapipe.framework.formats.landmark_pb2.NormalizedLandmarkList):
            The full set of facial landmarks detected by MediaPipe Face Mesh.
        image_w (int): Width of the image in pixels.
        image_h (int): Height of the image in pixels.

    Returns:
        tuple: (left_eye_center (ndarray), right_eye_center (ndarray)).
    """

    # MediaPipe: left eye 468–473, right eye 473–478
    LEFT_IRIS = [469, 470, 471, 472]
    RIGHT_IRIS = [474, 475, 476, 477]

    def avg_landmarks(indices):
        pts = [(landmarks.landmark[i].x * image_w,
                landmarks.landmark[i].y * image_h) for i in indices]
        pts = np.array(pts)
        return np.mean(pts, axis=0)

    left_eye_center = avg_landmarks(LEFT_IRIS)
    right_eye_center = avg_landmarks(RIGHT_IRIS)

    return left_eye_center, right_eye_center


def align_face(image):
    """
    Detects a face, computes eye centers, rotates the image so eyes are horizontal,
    and returns the aligned image.

    Args:
        image (np.ndarray): BGR image loaded by cv2

    Returns:
        tuple: (aligned_image (np.ndarray), angle (float), eye_centers (tuple)).

    Raises:
        ValueError: if no face detected
    """

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            raise ValueError("No face detected in the image.")
        
        # Take the first (and only) face since max_num_faces=1
        landmarks = results.multi_face_landmarks[0]

        # Find eye centers
        left_eye, right_eye = _compute_eye_centers(landmarks, w, h)

        # Compute rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))

        # Rotate image so that eyes are horizontal
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aligned = cv2.warpAffine(image, rotation_matrix, (w, h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)

        return aligned, angle, (left_eye, right_eye)
    

if __name__ == "__main__":
    img = cv2.imread("data/test/IMG_2817.png")
    aligned, angle, eyes = align_face(img)

    print("Rotation:", angle)
    cv2.imwrite("data/test/2817_aligned.png", aligned)
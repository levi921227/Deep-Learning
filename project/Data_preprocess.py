1import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet


# 進行臉部對齊
def align_face(img, keypoints):
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # 計算兩眼之間的角度
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))  # 計算旋轉角度

    # 計算旋轉中心為左眼與右眼的中點
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # 生成旋轉矩陣
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    # 將圖像旋轉對齊
    aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_img


def preprocess_and_align(image_path):
    img = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(img)

    aligned_faces = []
    for face in faces:
        keypoints = face['keypoints']
        aligned_img = align_face(img, keypoints)

        # 進一步裁剪和縮放
        x, y, width, height = face['box']
        face_img = aligned_img[y:y + height, x:x + width]
        face_resized = cv2.resize(face_img, (160, 160))

        face_resized = face_resized / 255.0
        cv2.imshow("Face", face_resized)
        cv2.waitKey(0)
        aligned_faces.append(face_resized)


    return np.array(aligned_faces)

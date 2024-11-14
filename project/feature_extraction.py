from keras_facenet import FaceNet
from Data_preprocess import preprocess_and_align, align_face


def generate_embeddings(face_img):
    embedder = FaceNet()
    embeddings = embedder.embeddings(face_img)
    return embeddings


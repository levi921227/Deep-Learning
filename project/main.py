import os
import numpy as np
from keras_facenet import FaceNet
from Data_preprocess import align_face, preprocess_and_align
from feature_extracion import generate_embeddings

# Path to the folder containing multiple jpg images
data_folder = "D:/Desktop/Pic/data/"

# Loop through all jpg files in the folder
for file_name in os.listdir(data_folder):
    if file_name.endswith(".jpg"):
        file_path = os.path.join(data_folder, file_name)

        # Preprocess and align faces from each image
        aligned_faces = preprocess_and_align(file_path)

        # Generate embeddings for each detected face
        for face in aligned_faces:
            face = np.expand_dims(face, axis=0)
            embeddings = generate_embeddings(face)
            print(f"Embeddings for {file_name}: {embeddings}")

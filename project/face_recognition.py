import os
import numpy as np
import random
from keras_facenet import FaceNet
from Data_preprocess import preprocess_and_align
from feature_extracion import generate_embeddings

# Path to the folder containing multiple jpg images organized by person and role (anchor, positive, negative)
data_folder = "D:/Desktop/Pic/data/"

# Instantiate FaceNet model
embedder = FaceNet()


# Function to process images and generate embeddings
def process_image(img_path):
    # Preprocess and align faces from the image
    aligned_faces = preprocess_and_align(img_path)
    embeddings = []

    # Generate embeddings for each detected face
    for face in aligned_faces:
        face = np.expand_dims(face, axis=0)
        embedding = generate_embeddings(face)
        embeddings.append(embedding)
    return embeddings


# Generate triplets with embeddings for training
def generate_triplets(data_folder):
    people = os.listdir(data_folder)
    triplets = []

    for person in people:
        person_path = os.path.join(data_folder, person)

        # 確保 anchor 和 positive 資料夾存在且有內容
        anchor_dir = os.path.join(person_path, 'anchor')
        print(anchor_dir)
        positive_dir = os.path.join(person_path, 'positive')
        print(positive_dir)
        if not os.path.exists(anchor_dir) or not os.path.exists(positive_dir):
            continue

        anchor_images = os.listdir(anchor_dir)
        positive_images = os.listdir(positive_dir)
        if not anchor_images or not positive_images:
            continue

        # 選擇 anchor 和 positive 圖片
        anchor_img_path = os.path.join(anchor_dir, random.choice(anchor_images))
        print(anchor_img_path)
        positive_img_path = os.path.join(positive_dir, random.choice(positive_images))
        print(positive_img_path)

        negative_dir = os.path.join(data_folder, person, 'negative')
        print(negative_dir)
        if not os.path.exists(negative_dir):  # 確保 negative 資料夾存在
            continue

        negative_images = os.listdir(negative_dir)
        if not negative_images:  # 確保 negative 資料夾有內容
            continue

        negative_img_path = os.path.join(negative_dir, random.choice(negative_images))

        # 生成三元組嵌入向量
        anchor_embeddings = process_image(anchor_img_path)
        positive_embeddings = process_image(positive_img_path)
        negative_embeddings = process_image(negative_img_path)

        # 儲存結果
        if anchor_embeddings and positive_embeddings and negative_embeddings:
            triplets.append((anchor_embeddings[0], positive_embeddings[0], negative_embeddings[0]))

    return triplets

# Main loop to generate triplets and print embeddings
triplets = generate_triplets(data_folder)
for i, (anchor, positive, negative) in enumerate(triplets):
    print(f"Triplet {i + 1}:")
    print("Anchor embedding:", anchor)
    print("Positive embedding:", positive)
    print("Negative embedding:", negative)


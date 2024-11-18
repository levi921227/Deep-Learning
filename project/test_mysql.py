# 測試是否能將產生的triples嵌入向量寫入mysql database 中
# database name：triplets_db ； table name : triplets
import os
import numpy as np
import random
import pickle
import mysql.connector
import tensorflow as tf
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

        # negative_person = random.choice(other_people)
        negative_dir = os.path.join(data_folder, person, 'negative')
        print(negative_dir)
        if not os.path.exists(negative_dir):  # 確保 negative 資料夾存在
            continue

        negative_images = os.listdir(negative_dir)
        if not negative_images:
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


def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, :512], y_pred[:, 512:1024], y_pred[:, 1024:]
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)


# Main loop to generate triplets and print embedding

# 儲存三元組到文件
#with open("triplets.pkl", "wb") as f:
    #pickle.dump(triplets, f)
#print("Triplets saved to triplets.pkl")

def connect_to_db():
    connection = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="1234",
        database="triplets_db"
    )
    return connection


def save_triplets_to_db(triplets):
    connection = connect_to_db()
    cursor = connection.cursor()

    insert_query = """
    INSERT INTO triplets (anchor_embedding, positive_embedding, negative_embedding)
    VALUES (%s, %s, %s)
    """
    for anchor, positive, negative in triplets:
        # 使用 pickle 序列化 numpy 数组
        anchor_blob = pickle.dumps(anchor)
        positive_blob = pickle.dumps(positive)
        negative_blob = pickle.dumps(negative)

        cursor.execute(insert_query, (anchor_blob, positive_blob, negative_blob))

    connection.commit()
    cursor.close()
    connection.close()


triplets = generate_triplets(data_folder)
print(f"Generated {len(triplets)} triplets.")
save_triplets_to_db(triplets)
print("Triplets saved to MySQL database.")






import os
import numpy as np
import mysql.connector
import keras.models
from Data_preprocess import preprocess_image


class FaceRecognitionDatabase:
    def __init__(self, model_path, db_config):
        # 加載模型
        self.model = keras.models.load_model(model_path)

        # 連接MySQL
        self.conn = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor()

        # 創建人臉特徵表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                embedding BLOB,
                image_path VARCHAR(255)
            )
        ''')
        self.conn.commit()

    def generate_embedding(self, image_path):
        """生成人臉嵌入向量"""
        face = preprocess_image(image_path)
        face_batch = np.expand_dims(face, axis=0)
        embedding = self.model.predict(face_batch)[0]
        return embedding

    def store_face_embedding(self, name, image_path):
        """將人臉特徵存入資料庫"""
        embedding = self.generate_embedding(image_path)
        embedding_bytes = embedding.tobytes()

        self.cursor.execute('''
            INSERT INTO face_embeddings (name, embedding, image_path) 
            VALUES (%s, %s, %s)
        ''', (name, embedding_bytes, image_path))
        self.conn.commit()

    def find_matching_face(self, input_image_path, threshold=0.7):
        """在資料庫中尋找相似人臉"""
        input_embedding = self.generate_embedding(input_image_path)

        # 查詢所有已存儲的嵌入向量
        self.cursor.execute('SELECT id, name, embedding, image_path FROM face_embeddings')
        stored_faces = self.cursor.fetchall()

        best_match = None
        best_similarity = 0

        for face_id, name, embedding_bytes, image_path in stored_faces:
            # 還原嵌入向量
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # 計算相似度
            similarity = np.dot(input_embedding, stored_embedding)

            if similarity > best_similarity and similarity > threshold:
                best_match = {
                    'id': face_id,
                    'name': name,
                    'similarity': similarity,
                    'image_path': image_path
                }
                best_similarity = similarity

        return best_match

    def close_connection(self):
        """關閉資料庫連接"""
        self.cursor.close()
        self.conn.close()


def main():
    # 配置
    model_path = r"C:\Users\User\PycharmProjects\Face\Project\Final Project\face_recognition_models\refined_embedding_model.h5"
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '1234',
        'database': 'face_recognition_db'
    }

    # 初始化
    face_db = FaceRecognitionDatabase(model_path, db_config)

    try:
        # 範例：存儲新人臉
        face_db.store_face_embedding('chaewon', 'D:/Desktop/Pic/data/anchor05.jpg')

        # 範例：識別人臉
        match = face_db.find_matching_face('D:/Desktop/Pic/data/anchor07.jpg')

        if match:
            print(f"找到相似人臉: {match['name']}")
            print(f"相似度: {match['similarity']:.4f}")
        else:
            print("未找到匹配人臉")

    finally:
        face_db.close_connection()


if __name__ == "__main__":
    main()











from tkinter import Tk, Label, Button, Frame, filedialog, messagebox
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk  # 用於顯示圖片

class FaceRecognitionApp:
    def __init__(self, root, face_db):
        self.root = root
        self.face_db = face_db
        self.root.title("人臉辨識系統")
        self.root.geometry("700x500")

        # 上方顯示區域
        self.display_frame = Frame(root)
        self.display_frame.pack(pady=10)
        
        # 介面元件
        self.label = Label(root, text="上傳圖片以進行辨識")
        self.label.pack(pady=20)

        self.image_label = Label(self.display_frame, text="(上傳圖片會顯示在此處)", width=40, height=10, bg="gray")
        self.image_label.pack(pady=10)

        self.button_frame = Frame(root)
        self.button_frame.pack(pady=10)

        self.upload_button = Button(self.button_frame, text="上傳圖片", command=self.upload_image, width=15)
        self.upload_button.grid(row=0, column=0, padx=5)

        self.reset_button = Button(self.button_frame, text="重置", command=self.reset, width=15)
        self.reset_button.grid(row=0, column=1, padx=5)

        self.result_label = Label(root, text="", fg="blue")
        self.result_label.pack(pady=20)

        self.progress = Progressbar(root, orient="horizontal", length=300, mode="indeterminate")
        
    def upload_image(self):
        """用戶上傳圖片並執行人臉辨識"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )

        if file_path:
            self.progress.pack(pady=10)
            self.progress.start()
            try:
                img = Image.open(file_path)
                img.thumbnail((600, 400))
                img_tk = ImageTk.PhotoImage(img)
                self.image_label.config(image=img_tk, text="")
                self.image_label.image = img_tk

                # 確認是否進行辨識
                confirm = messagebox.askyesno("確認", "是否要繼續進行人臉辨識？")
                if confirm:
                    # 顯示加載動畫並執行辨識
                    self.progress.pack(pady=10)
                    self.progress.start()
                    self.root.after(1000, lambda: self.recognize_face(file_path))  # 模擬處理延遲
                else:
                    self.result_label.config(text="已取消人臉辨識。")
                    
                match = self.face_db.find_matching_face(file_path)

                if match:
                    self.result_label.config(
                        text=f"找到相似人臉: {match['name']}\n相似度: {match['similarity']:.4f}"
                    )
                else:
                    self.result_label.config(text="未找到匹配人臉")
            except Exception as e:
                messagebox.showerror("錯誤", f"處理圖片時發生錯誤：{e}")


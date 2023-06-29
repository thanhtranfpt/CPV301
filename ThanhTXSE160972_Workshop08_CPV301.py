def EigenFacesAlgorithm(test_img,
                        train_image_folder_path = r"K:\LEARN AI FPT\PROJECTS\images\AI1707_sampleFaces\ASI\ASI"):
    import cv2
    import os
    import numpy as np

    # Chuẩn bị dữ liệu
    dataset_path = train_image_folder_path
    face_images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            # label = int(filename.split(".")[0])
            label = str(filename.split(".")[0])
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_images.append(image)
            labels.append(label)

    # Tiền xử lý
    face_images = [cv2.resize(image, (100, 100)) for image in face_images]

    # Phân tích thành phần chính (PCA)
    X = np.array(face_images)
    X = X.reshape(X.shape[0], -1)
    mean, eigen_vectors = cv2.PCACompute(X, mean=None, maxComponents=10)

    # Xây dựng Eigenfaces
    eigen_faces = eigen_vectors.reshape(-1, 100, 100)

    # Huấn luyện mô hình
    face_space = np.dot(X - mean, eigen_vectors.T)
    face_labels = np.array(labels)

    # Nhận dạng khuôn mặt
    '''test_image = cv2.imread(r"K:\LEARN AI FPT\PROJECTS\images\AI1707_sampleFaces\numbersASI\ASI\6.1.jpg",
                            cv2.IMREAD_GRAYSCALE)'''
    test_image = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (100, 100))
    test_image_vector = test_image.reshape(1, -1)
    test_image_space = np.dot(test_image_vector - mean, eigen_vectors.T)
    distances = np.linalg.norm(face_space - test_image_space, axis=1)
    min_distance_index = np.argmin(distances)
    predicted_label = face_labels[min_distance_index]
    #print(predicted_label)

    return predicted_label

import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

class LabeledImage(tk.Frame):
    def __init__(self, master=None, title="", **kwargs):
        super().__init__(master, **kwargs)
        self.title_label = tk.Label(self, text=title, font=("Helvetica", 12))
        self.title_label.pack(side="top", padx=10, pady=5)
        self.img_label = tk.Label(self)
        self.img_label.pack(side="top", padx=10, pady=5)

        # Load placeholder image
        self.placeholder_img = Image.new('RGB', (700, 500), color='gray')
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_img)

    def set_image(self, img_path):
        if img_path:
            img = Image.open(img_path)
            img = img.resize((700, 500))
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text=img_path)
            self.img_label.image = photo
        else:
            self.img_label.config(image=self.placeholder_photo, text="")
            self.img_label.image = self.placeholder_photo

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.img_paths = [None, None]
        self.img_titles = ["Choose an image to recognize face"]
        self.create_widgets()

    def create_widgets(self):
        # add title label
        self.title_label = tk.Label(self.master, text="Object recognition, face recognition : Eigenfaces algorithm", font=("Helvetica", 16))
        self.title_label.grid(row=0, column=0, pady=10)

        self.labeled_imgs = []
        for i in range(1):
            labeled_img = LabeledImage(self.master, title=self.img_titles[i])
            labeled_img.grid(row=1, column=i, padx=10)

            labeled_img.set_image(None)
            self.labeled_imgs.append(labeled_img)

        self.browse_buttons = []
        for i in range(1):
            self.browse_buttons.append(tk.Button(self.master, text="Browse", command=lambda idx=i: self.browse_files(idx)))
            self.browse_buttons[i].grid(row=2, column=i, padx=10)

        for i in [1]:
            self.browse_buttons.append(tk.Button(self.master, text="Browse training images folder", command=lambda idx=i: self.browse_folder(idx)))
            self.browse_buttons[i].grid(row=3, column=0, padx=10)

        self.show_button = tk.Button(self.master, text="Recogniton", command=self.show_images)
        self.show_button.grid(row=4, column=0, pady=10)

    def browse_folder(self,idx):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.img_paths[idx] = folder_path

    def browse_files(self, idx):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.gif")])
        if file_path:
            self.labeled_imgs[idx].set_image(file_path)
            self.img_paths[idx] = file_path

    def show_images(self):
        if self.img_paths[0] is not None and self.img_paths[1] is not None:
            test_img = cv2.imread(self.img_paths[0])
            predicted_label = EigenFacesAlgorithm(test_img,self.img_paths[1])

            my_label = tk.Label(root, text=' '*10 +'Predicted: '+predicted_label + ' '*10)
            my_label.grid(row=6, column=0)

root = tk.Tk()
root.title("Workshop 8 - Trần Xuân Thành")
app = Application(master=root)
app.mainloop()

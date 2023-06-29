import tkinter as tk
from tkinter import filedialog

import cv2_ext
from PIL import Image, ImageTk

#=============================================================
from General_Functions import HaarCascade_face_detection
from General_Functions import EigenFacesAlgorithm

def Workshop8(image_path,train_folder_path):
    import cv2
    img = cv2.imread(image_path)
    marked_img, face_locations, faces = HaarCascade_face_detection(img,showResult=False)
    names = []
    for face in faces:
        predicted = EigenFacesAlgorithm(face,train_folder_path)
        names.append(predicted)

    # Check if true:
    can_check = None
    if len(names) == 1:
        real_name = image_path.split("/")[-1].split('.')[0]
        #print(real_name)
        #print(names[0])
        if real_name == names[0]:
            print('True')
            can_check = True
        else:
            print('False')
            can_check = False

    print('Number of faces: ',len(faces))
    return names, face_locations, marked_img, can_check
####################################################################
class LabeledImage(tk.Frame):
    def __init__(self, master=None, title="", **kwargs):
        super().__init__(master, **kwargs)
        self.title_label = tk.Label(self, text=title, font=("Helvetica", 12))
        self.title_label.pack(side="top", padx=10, pady=5)
        self.img_label = tk.Label(self)
        self.img_label.pack(side="top", padx=10, pady=5)

        # Load placeholder image
        self.placeholder_img = Image.new('RGB', (500, 400), color='gray')
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_img)

    def set_image(self, img_path):
        if img_path:
            img = Image.open(img_path)
            img = img.resize((500, 400))
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
        self.img_titles = ["Choose an image to recognize face", "Faces recognized in the image"]
        self.create_widgets()

    def create_widgets(self):
        # add title label
        self.title_label = tk.Label(self.master, text="Object recognition, face recognition :", font=("Helvetica", 16))
        self.title_label.grid(row=0, column=0, pady=10)
        self.title_label = tk.Label(self.master, text="Eigenfaces algorithm", font=("Helvetica", 16))
        self.title_label.grid(row=0, column=1, pady=10)

        self.labeled_imgs = []
        for i in range(2):
            labeled_img = LabeledImage(self.master, title=self.img_titles[i])
            labeled_img.grid(row=1, column=i, padx=10)

            labeled_img.set_image(None)
            self.labeled_imgs.append(labeled_img)

        self.browse_buttons = []
        for i in range(1):
            self.browse_buttons.append(tk.Button(self.master, text="Browse for an image", command=lambda idx=i: self.browse_files(idx)))
            self.browse_buttons[i].grid(row=2, column=i, padx=10)

        for i in range(1,2):
            self.browse_buttons.append(tk.Button(self.master, text="Browse for the the folder used for training", command=lambda idx=i: self.browse_folder(idx)))
            self.browse_buttons[i].grid(row=3, column=0, padx=10)

        self.show_button = tk.Button(self.master, text="Show Result", command=self.show_images)
        self.show_button.grid(row=3, column=1, pady=10)

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
            predicted, face_locations, marked_img, can_check = Workshop8(self.img_paths[0], self.img_paths[1])
            cv2_ext.imwrite('temp.png',marked_img)
            names = ''
            for name in predicted:
                names += name + ', '
            my_label = tk.Label(root, text=' ' * 30 + 'Predicted: ' + names + ' ' * 30)
            my_label.grid(row=5, column=1)

            my_label = tk.Label(root, text=' ' * 30 + ' ' + ' ' * 30)
            my_label.grid(row=6, column=1)
            if can_check is not None:
                my_label = tk.Label(root, text=' ' * 30 + 'Check: ' + str(can_check) + ' ' * 30)
                my_label.grid(row=6, column=1)
            self.labeled_imgs[1].set_image('temp.png')
            #self.img_paths[2] = 'temp.png'

root = tk.Tk()
root.title("Workshop 8: Trần Xuân Thành - updated")
app = Application(master=root)
app.mainloop()

import tkinter as tk
from tkinter import filedialog

import cv2_ext
import cv2
from PIL import Image, ImageTk

#----------------------------------------------------------------------
def HaarCascade_face_detection(img, showResult = True,
                    model_file = r"K:\LEARN AI FPT\PROJECTS\haarcascade_frontalface_default.xml",
                    scaleFactor=1.1, minNeighbors=5,
                    marked_color = (0, 255, 0)) :
    import cv2

    # Load the input image
    #img = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade classifier model
    face_cascade = cv2.CascadeClassifier(model_file)

    # Use the classifier model to detect faces in the input image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw rectangles around the detected faces in the input image
    face_locations = []
    marked_img = img.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(marked_img, (x, y), (x+w, y+h), marked_color, 2)
        face_locations.append(((x, y), (x+w, y+h)))

    # Display the output image
    if showResult == True:
        cv2.imshow('Face detection', marked_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    # Store faces into a list:
    faces = []
    for faceLoc in face_locations:
        x1,y1 = faceLoc[0]
        x2,y2 = faceLoc[1]
        face = img[y1:y2, x1:x2]
        faces.append(face)


    return marked_img, face_locations, faces


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
        self.img_titles = ["Input Image", "Faces in the Input Image"]
        self.create_widgets()

    def create_widgets(self):
        # add title label
        self.title_label = tk.Label(self.master, text="Object detection", font=("Helvetica", 16))
        self.title_label.grid(row=0, column=0, pady=10)
        self.title_label = tk.Label(self.master, text=" - Face detection",font=("Helvetica", 16))
        self.title_label.grid(row=0, column=1, pady=10)

        self.labeled_imgs = []
        for i in range(2):
            labeled_img = LabeledImage(self.master, title=self.img_titles[i])
            labeled_img.grid(row=1, column=i, padx=10)

            labeled_img.set_image(None)
            self.labeled_imgs.append(labeled_img)

        self.browse_buttons = []
        for i in range(1):
            self.browse_buttons.append(tk.Button(self.master, text="Browse", command=lambda idx=i: self.browse_files(idx)))
            self.browse_buttons[i].grid(row=2, column=i, padx=10)

        self.show_button = tk.Button(self.master, text="Detect Faces", command=self.show_images)
        self.show_button.grid(row=3, column=1, pady=10)

    def browse_files(self, idx):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.gif")])
        if file_path:
            self.labeled_imgs[idx].set_image(file_path)
            self.img_paths[idx] = file_path

    def show_images(self):
        if self.img_paths[0] is not None:
            img = cv2.imread(self.img_paths[0])
            marked_image, _, __ = HaarCascade_face_detection(img,showResult=False)
            marked_image_path = 'temp.png'
            cv2_ext.imwrite(marked_image_path,marked_image)
            self.labeled_imgs[1].set_image(marked_image_path)
            self.img_paths[1] = marked_image_path

root = tk.Tk()
root.title("Workshop 7 - Trần Xuân Thành")
app = Application(master=root)
app.mainloop()

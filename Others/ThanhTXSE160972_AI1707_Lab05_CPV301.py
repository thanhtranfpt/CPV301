import tkinter as tk
from tkinter import filedialog

import cv2_ext
from PIL import Image, ImageTk


def RANSAC_Alignment(ReferenceImage_path, TestImage_path,
                     showResult=True,
                     Lowes_ratio=0.7,
                     RANSAC_distance_threshold=10):
    import cv2
    import numpy as np

    # Load the two images to be aligned
    img1 = cv2.imread(ReferenceImage_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(TestImage_path, cv2.IMREAD_GRAYSCALE)

    # Create the SIFT detector object
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Create a Brute-Force Matcher object
    bf = cv2.BFMatcher()

    # Match descriptors from the two images
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply Lowe's ratio test to filter out good matches
    good_matches = []
    for m, n in matches:
        if m.distance < Lowes_ratio * n.distance:
            good_matches.append(m)

    # Set the threshold distance for RANSAC
    threshold = RANSAC_distance_threshold

    # Create empty arrays to store the keypoints for RANSAC
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Run RANSAC to estimate the homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold)

    # Warp the second image using the estimated homography
    aligned_img = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

    # Display the aligned image
    if showResult == True:
        cv2.imshow('Reference Image', ReferenceImage_path)
        cv2.imshow('Test Image', TestImage_path)
        cv2.imshow('Aligned Image', aligned_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return aligned_img


def Image_Alignment_RANSAC(ReferenceImage_path, TestImage_path,
                           showResult=True,
                           Lowes_ratio=0.7,
                           good_matches_length=10):
    import cv2
    import numpy as np

    source_image = cv2.imread(ReferenceImage_path)
    source_image_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)

    test_image = cv2.imread(TestImage_path)
    test_image_gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Calculate the homography matrix to align the test image with the source image
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(source_image_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(test_image_gray, None)
    matcher = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter out false matches
    good_matches = []
    for m, n in matches:
        if m.distance < Lowes_ratio * n.distance:
            good_matches.append(m)

    # Calculate the homography matrix using RANSAC
    if len(good_matches) > good_matches_length:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        h, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)

        # Warp the test image using the homography matrix
        aligned_image = cv2.warpPerspective(test_image, h, (source_image.shape[1], source_image.shape[0]))

        # Display the aligned image in a window
        if showResult == True:
            cv2.imshow('Reference Image', source_image)
            cv2.imshow('Test Image', test_image)
            cv2.imshow('Aligned Image', aligned_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    return aligned_image

class LabeledImage(tk.Frame):
    def __init__(self, master=None, title="", **kwargs):
        super().__init__(master, **kwargs)
        self.title_label = tk.Label(self, text=title, font=("Helvetica", 12))
        self.title_label.pack(side="top", padx=10, pady=5)
        self.img_label = tk.Label(self)
        self.img_label.pack(side="top", padx=10, pady=5)

        # Load placeholder image
        self.placeholder_img = Image.new('RGB', (250, 400), color='gray')
        self.placeholder_photo = ImageTk.PhotoImage(self.placeholder_img)

    def set_image(self, img_path):
        if img_path:
            img = Image.open(img_path)
            img = img.resize((250, 400))
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
        self.img_paths = [None, None, None, None]
        self.img_titles = ["Reference Image", "Test Image", "RANSAC applied", "Aligned Image"]
        self.create_widgets()

    def create_widgets(self):
        # add title label
        self.title_label = tk.Label(self.master, text="RANSAC algorithm for", font=("Helvetica", 16))
        self.title_label.grid(row=0, column=1, pady=10)
        self.title_label = tk.Label(self.master, text="Image alignment based features",font=("Helvetica", 16))
        self.title_label.grid(row=0, column=2, pady=10)

        self.labeled_imgs = []
        for i in range(4):
            labeled_img = LabeledImage(self.master, title=self.img_titles[i])
            labeled_img.grid(row=1, column=i, padx=10)

            labeled_img.set_image(None)
            self.labeled_imgs.append(labeled_img)

        self.browse_buttons = []
        for i in range(2):
            self.browse_buttons.append(tk.Button(self.master, text="Browse", command=lambda idx=i: self.browse_files(idx)))
            self.browse_buttons[i].grid(row=2, column=i, padx=10)

        self.show_button = tk.Button(self.master, text="Show Result", command=self.show_images)
        self.show_button.grid(row=3, column=2, pady=10)

    def browse_files(self, idx):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.gif")])
        if file_path:
            self.labeled_imgs[idx].set_image(file_path)
            self.img_paths[idx] = file_path

    def show_images(self):
        if self.img_paths[0] is not None and self.img_paths[1] is not None:
            # RANSAC applied
            RANSAC_applied = RANSAC_Alignment(self.img_paths[0],self.img_paths[1],showResult=False)
            RANSAC_applied_path = 'RANSAC_applied_image.png'
            cv2_ext.imwrite(RANSAC_applied_path, RANSAC_applied)
            self.labeled_imgs[2].set_image(RANSAC_applied_path)
            self.img_paths[2] = RANSAC_applied_path
            # Image Alignment
            aligned_image = Image_Alignment_RANSAC(self.img_paths[0],self.img_paths[1],showResult=False)
            aligned_image_path = r'aligned_RANSAC_image.png'
            cv2_ext.imwrite(aligned_image_path,aligned_image)
            self.labeled_imgs[3].set_image(aligned_image_path)
            self.img_paths[3] = aligned_image_path

root = tk.Tk()
root.title("Workshop 5 - Trần Xuân Thành")
app = Application(master=root)
app.mainloop()

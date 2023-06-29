import tkinter as tk
from tkinter import filedialog

import cv2_ext
from PIL import Image, ImageTk

#----------------------------------------------------------------------
def Image_Matching(image1_path, image2_path, ratio=0.75, showResult = True):
    import cv2
    import numpy as np

    # Read input images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Create a SIFT object to detect keypoints and extract descriptors
    sift = cv2.SIFT_create()

    # Detect keypoints and extract descriptors from the input images
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Create a BFMatcher object to match the descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to select good matches
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    # Draw the matched keypoints between the two images
    result = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the result
    if showResult == True:
        cv2.imshow('Image Matching Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return result

#----------------------------------------------------------------------
def ImageStitching(left_image_path, right_image_path,
                   showResult = True,
                   ratio = 0.75, matches_length = 200):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import imageio.v2

    # def to plot images by matplotlib
    def plot_img(img, size=(7, 7), title=""):
        cmap = "gray" if len(img.shape) == 2 else None
        plt.figure(figsize=size)
        plt.imshow(img, cmap=cmap)
        plt.suptitle(title)
        plt.show()

    def plot_imgs(imgs, cols=5, size=7, title=""):
        rows = len(imgs) // cols + 1
        fig = plt.figure(figsize=(cols * size, rows * size))
        for i, img in enumerate(imgs):
            cmap = "gray" if len(img.shape) == 2 else None
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(img, cmap=cmap)
        plt.suptitle(title)
        plt.show()


    # Load two images
    left_img = imageio.v2.imread(left_image_path)
    right_img = imageio.v2.imread(right_image_path)

    if showResult == True: plot_imgs([left_img, right_img], size=8)

    # Convert images to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_RGB2GRAY)

    # Find keypoints and descriptors using SIFT
    SIFT_detector = sift = cv2.SIFT_create()
    kp1, des1 = SIFT_detector.detectAndCompute(right_gray, None)
    kp2, des2 = SIFT_detector.detectAndCompute(left_gray, None)

    # Match keypoints using Brute-Force Matcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Brute-Force KNN returns a list of k candidates for each keypoint
    rawMatches = bf.knnMatch(des1, des2, 2)

    # Filter matches using Lowe's ratio test
    matches = []
    #ratio = 0.75
    for m, n in rawMatches:
        # retain keypoint pairs such that for kp1, the distance between kp1 and candidate 1 is much smaller than the distance between kp1 and candidate 2
        if m.distance < n.distance * ratio:
            matches.append(m)

    # Because there are thousands of match keypoints, we only take the best 100 -> 200 pairs for faster processing speed
    matches = sorted(matches, key=lambda x: x.distance, reverse=True)
    #matches_length = 200
    matches = matches[:matches_length]

    img3 = cv2.drawMatches(right_img, kp1, left_img, kp2, matches, None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Save the img3 image
    matching_images_stitching = Image_Matching(left_image_path,right_image_path,showResult=False)
    if showResult == True: cv2.imshow('Images Matching', matching_images_stitching)

    kp1 = np.float32([kp.pt for kp in kp1])
    kp2 = np.float32([kp.pt for kp in kp2])
    pts1 = np.float32([kp1[m.queryIdx] for m in matches])
    pts2 = np.float32([kp2[m.trainIdx] for m in matches])

    # Calculate homography matrix using RANSAC
    (H, status) = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    h1, w1 = right_img.shape[:2]
    h2, w2 = left_img.shape[:2]

    # Warp the second image onto the first image using the homography matrix
    result = cv2.warpPerspective(right_img, H, (w1 + w2, h1))

    # Merge the two images
    result[0:h2, 0:w2] = left_img

    # Save the result
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result_image.png", result)

    # Show the result
    result = cv2.imread("result_image.png")
    if showResult == True:
        cv2.imshow('Result Image Preview', cv2.resize(result, (750, 750)))
        cv2.waitKey()
        cv2.destroyAllWindows()


    return result, matching_images_stitching
#----------------------------------------------------------------------
#----------------------------------------------------------------------

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

    def set_image_single(self, img_path):
        if img_path:
            img = Image.open(img_path)
            img = img.resize((150, 400))
            photo = ImageTk.PhotoImage(img)
            self.img_label.config(image=photo, text=img_path)
            self.img_label.image = photo
        else:
            self.img_label.config(image=self.placeholder_photo, text="")
            self.img_label.image = self.placeholder_photo
    def set_image_double(self, img_path):
        if img_path:
            img = Image.open(img_path)
            img = img.resize((300, 400))
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
        self.img_titles = ["Left Image", "Right Image", "Images Matching", "Images Stitching"]
        self.create_widgets()

    def create_widgets(self):
        # add title label
        self.title_label = tk.Label(self.master, text="Motion models", font=("Helvetica", 16))
        self.title_label.grid(row=0, column=1, pady=10)
        self.title_label = tk.Label(self.master, text="- Image stitching",font=("Helvetica", 16))
        self.title_label.grid(row=0, column=2, pady=10)

        self.labeled_imgs = []
        for i in range(2):
            labeled_img = LabeledImage(self.master, title=self.img_titles[i])
            labeled_img.grid(row=1, column=i, padx=10)

            labeled_img.set_image_single(None)
            self.labeled_imgs.append(labeled_img)
        for i in range(2,4):
            labeled_img = LabeledImage(self.master, title=self.img_titles[i])
            labeled_img.grid(row=1, column=i, padx=10)

            labeled_img.set_image_double(None)
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
            self.labeled_imgs[idx].set_image_single(file_path)
            self.img_paths[idx] = file_path

    def show_images(self):
        if self.img_paths[0] is not None and self.img_paths[1] is not None:
            stitched_image, matching_image = ImageStitching(self.img_paths[0], self.img_paths[1], showResult=False)
            # Matching images
            matching_image_path = 'temp001.png'
            cv2_ext.imwrite(matching_image_path, matching_image)
            self.labeled_imgs[2].set_image_double(matching_image_path)
            self.img_paths[2] = matching_image_path
            # Stiching images
            stiched_image_path = 'temp002.png'
            cv2_ext.imwrite(stiched_image_path,stitched_image)
            self.labeled_imgs[3].set_image_double(stiched_image_path)
            self.img_paths[3] = stiched_image_path

root = tk.Tk()
root.title("Workshop 6 - Trần Xuân Thành")
app = Application(master=root)
app.mainloop()

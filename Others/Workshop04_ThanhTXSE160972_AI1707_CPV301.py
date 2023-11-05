def image_kMeanCluster(image_path, k):
    from sklearn.cluster import KMeans
    import numpy as np
    import cv2

    img = cv2.imread(image_path)
    img_flat = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(img_flat)
    img_flat2 = img_flat.copy()

    # loops for each cluster center
    for i in np.unique(kmeans.labels_):
        img_flat2[kmeans.labels_ == i, :] = kmeans.cluster_centers_[i]

    img2 = np.reshape(img_flat2, img.shape)

    return img2, kmeans.inertia_

def Elbow_kMeanCluster(image_path, K_list):
    import matplotlib.pyplot as plt
    import cv2

    # k_vals = list(range(2, 21, 2))
    k_vals = K_list
    img_list = []
    inertia = []
    for k in k_vals:
        # print(k)
        img2, ine = image_kMeanCluster(image_path, k)
        img_list.append(img2)
        inertia.append(ine)

    # Plot to find optimal number of clusters
    plt.plot(k_vals, inertia)
    plt.scatter(k_vals, inertia)
    plt.xlabel('k')
    plt.ylabel('Inertia')

    # Plot all segmented images:
    plt.figure(figsize=[10, 20])
    for i in range(len(k_vals)):
        plt.subplot(5, 2, i + 1)
        img_plt = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_plt)
        plt.title('k = ' + str(k_vals[i]))
        plt.axis('off')
    plt.show()

    return img_list, inertia


def MeanShift_segmentation(image_path, showResult = True,
                           kernel_size = 10, max_iteration = 50):
    import cv2

    # Load the input image
    img = cv2.imread(image_path)

    # Perform mean shift segmentation
    segmented_img = cv2.pyrMeanShiftFiltering(img, kernel_size, max_iteration)

    # This is not need, just used for count the number of clusters:
    # WARNING: NOT SURE
    # Convert the segmented image to grayscale
    gray = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)

    # Apply a binary threshold to the grayscale image to create a binary image
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the contours on the original image
    segmented_img2 = img.copy()
    cv2.drawContours(segmented_img2, contours, -1, (0, 255, 0), 2)

    # Display the original image with the contours
    if showResult == True:
        cv2.imshow('Contours', segmented_img2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Display the input and segmented images
    if showResult == True:
        cv2.imshow('Input Image', img)
        cv2.imshow('Segmented Image', segmented_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


    return segmented_img


def kMeansClustering(image_path, k, showResult = True):
    import cv2
    import numpy as np

    # Load the input image
    img = cv2.imread(image_path)

    # Reshape the image to a 2D array of pixels
    pixel_values = img.reshape((-1, 3))

    # Convert the data type to float32
    pixel_values = np.float32(pixel_values)

    # Define the termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Set the number of clusters


    # Perform k-means clustering
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert the center values back to uint8
    centers = np.uint8(centers)

    # Reshape the labels to the original image shape
    labels = labels.reshape((img.shape[0], img.shape[1]))

    # Apply the colors of the centers to the segmented image
    segmented_img = centers[labels]

    # Display the input and segmented images
    if showResult == True:
        cv2.imshow('Input Image', img)
        cv2.imshow('Segmented Image', segmented_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return segmented_img

# ----------------------------------------------------------------------
def Snake_segmentation(image_path, ROI=False, region=((50, 100), (200, 500)),
                       showResult=True, colored_mask=255,
                       params={'Gaussian': [(5, 5), 0],
                               'Canny': [100, 200],
                               'sigma': 0.5, 'k': 200, 'min_size': 100}):
    import cv2
    import numpy as np

    # Read in the input image
    img = cv2.imread(image_path)

    # Define the region of interest (ROI) coordinates
    if ROI == True:
        x1, y1 = region[0]
        x2, y2 = region[-1]

        # Extract the ROI from the image
        img = img[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a Gaussian filter
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Find edges using the Canny algorithm
    canny = cv2.Canny(blur, 100, 200)

    # Set parameters for the graph-based segmentation
    sigma = 0.5
    k = 200
    min_size = 100

    # Perform the graph-based segmentation
    segmenter = cv2.ximgproc.segmentation.createGraphSegmentation(sigma, k, min_size)
    segmentation = segmenter.processImage(canny)

    # Draw the segmentation result on the original image
    mask = np.zeros_like(canny)
    mask[segmentation == 0] = 255
    result = cv2.bitwise_and(img, img, mask=mask)

    # Show the result
    if showResult == True:
        cv2.imshow("Input image", img)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


def Watershed_segmentation(image_path, showResult=True,
                           colored_boundary=[255, 0, 0],
                           params={'threshold': 0.7}):
    import cv2
    import numpy as np

    # Read the image
    img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Apply watershed segmentation
    markers = cv2.connectedComponents(sure_fg)[1]
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    # Draw the watershed boundaries
    segmented_img = img.copy()
    segmented_img[markers == -1] = [255, 0, 0]

    # Display the result
    if showResult == True:
        cv2.imshow('Input image', img)
        cv2.imshow('Result', segmented_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return segmented_img


def F1(image_path):
    Snake_segmentation(image_path)
def F2(image_path):
    Watershed_segmentation(image_path)
def F3(image_path, k):
    kMeansClustering(image_path, k= k)
def F4(image_path):
    MeanShift_segmentation(image_path)

def printMenu():
    while True:
        impath = input('Enter image path : ')
        if len(impath) < 1:
            impath = r"K:\LEARN AI FPT\PROJECTS\images\hoa qua.JPG"
        print('*'*10,' MENU ','*'*10)
        print('1. Function 1: Snakes algorithm')
        print('2. Function 2:  watershed algorithm')
        print('3. Function 3: K Means')
        print('4. Function 4: Mean shift')
        print('5. EXIT')
        opt = int(input('Choose [1..5]: '))
        if opt < 1 or opt > 5:
            continue
        else:
            break

    return opt, impath

def main(opt):
    global impath
    if opt == 1:
        F1(impath)
    elif opt == 2:
        F2(impath)
    elif opt == 3:
        k = int(input('Enter number of clusters: k = '))
        F3(impath, k)
        print("Now let's show how K-Means applied at different values of K:")
        x = input('Enter start value, end value, step (for e.g: 2,10,2) : ')
        if len(x) < 1:
            Elbow_kMeanCluster(impath, range(2, 10, 2))
        else:
            x = [int(k) for k in x.strip().split(',')]
            Elbow_kMeanCluster(impath, range(x[0], x[1], x[2]))
    elif opt == 4:
        F4(impath)
    else:
        quit()

if __name__ == '__main__':
    while True:
        opt, impath = printMenu()
        main(opt)
        print('-'*20)
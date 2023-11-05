def HOG_transform(img, showResult=False, resize=(128, 64),
                  orientations=8, pixels_per_cell=(16, 16),
                  cells_per_block=(1, 1), visualize=True, channel_axis=-1,
                  in_range=(0, 10)):


    # --- img = cv2.imread(ipath): is 3-CHANNELS.
    import cv2

    image = cv2.resize(img, resize)

    import matplotlib.pyplot as plt

    from skimage.feature import hog
    from skimage import data, exposure

    fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                        cells_per_block=cells_per_block, visualize=visualize,
                        channel_axis=channel_axis)

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=in_range)

    if showResult == True:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.show()

    return hog_image_rescaled

#----------------------------------------------------------------------------------------------------
def Harris_Corner_Detector(image, showResult=False,
                           threshold_value = 0.01, marked_color = [0, 0, 255] ):

    # --- image = cv2.imread(ipath): is 3-CHANNELS.
    # Python program to illustrate corner detection with Harris Corner Detection Method

    # organizing imports
    import cv2
    import numpy as np

    # convert the input image into grayscale color space
    operatedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # modify the data type : setting to 32-bit floating point
    operatedImage = np.float32(operatedImage)

    # apply the cv2.cornerHarris method to detect the corners with appropriate values as input parameters
    dest = cv2.cornerHarris(operatedImage, 2, 5, 0.07)

    # Results are marked through the dilated corners
    dest = cv2.dilate(dest, None)

    # Reverting back to the original image, with optimal threshold value
    if showResult == True:
        cv2.imshow('dest', dest)

        marked_image = image.copy()
        marked_image[dest > threshold_value * dest.max()] = marked_color

        # the window showing output image with corners
        cv2.imshow('Image with Borders', marked_image)

        # De-allocate any associated memory usage
        cv2.waitKey()
        cv2.destroyAllWindows()


    return dest

# ----------------------------------------------------------------------------------------------------
def Canny_Edge_Detector(img, params=(50, 100)):
        new_img = cv2.Canny(img, params[0], params[1])
        return new_img

# ----------------------------------------------------------------------------------------------------
def Hough_Transform_easy(image, showResult=False, params_CannyEdge=(50, 150), apertureSize=3,
                            threshold=100, minLineLength=5, maxLineGap=10):
    # --- image = cv2.imread(ipath)
    import cv2
    import numpy as np

    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use canny edge detection
    edges = cv2.Canny(gray, params_CannyEdge[0], params_CannyEdge[1], apertureSize=apertureSize)

    # Apply HoughLinesP method to directly obtain line end points
    lines_list = []
    lines = cv2.HoughLinesP(
        edges,  # Input edge image
        1,  # Distance resolution in pixels
        np.pi / 180,  # Angle resolution in radians
        threshold=threshold,  # Min number of votes for valid line
        minLineLength=minLineLength,  # Min allowed length of line
        maxLineGap=maxLineGap  # Max allowed gap between line for joining them
    )

    # Iterate over points
    marked_image = image.copy()
    for points in lines:
        # Extracted points nested in the list
        x1, y1, x2, y2 = points[0]
        # Draw the lines joing the points: On the original image
        cv2.line(marked_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Maintain a simples lookup list for points
        lines_list.append([(x1, y1), (x2, y2)])

    # Save the result image
    if showResult == True:
        cv2.imshow('detectedLines.png', marked_image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    return marked_image, lines_list

# ----------------------------------------------------------------------------------------------------
def Hough_Transform_hard(img, showResult=False, params_CannyEdge=(50, 150), apertureSize=3,
                             marked_color=(0, 0, 255)):

    # --- img = cv2.imread(ipath)
    # Python program to illustrate HoughLine method for line detection
    import cv2
    import numpy as np

    # Convert the img to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply edge detection method on the image
    edges = cv2.Canny(gray, params_CannyEdge[0], params_CannyEdge[1], apertureSize=apertureSize)

    # This returns an array of r and theta values
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    marked_img = img.copy()
    lines_list = []
    # The below for loop runs till r and theta values are in the range of the 2d array
    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        # Stores the value of cos(theta) in a
        a = np.cos(theta)

        # Stores the value of sin(theta) in b
        b = np.sin(theta)

        # x0 stores the value rcos(theta)
        x0 = a * r

        # y0 stores the value rsin(theta)
        y0 = b * r

        # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
        x1 = int(x0 + 1000 * (-b))

        # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
        y1 = int(y0 + 1000 * (a))

        # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
        x2 = int(x0 - 1000 * (-b))

        # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
        y2 = int(y0 - 1000 * (a))

        # cv2.line draws a line in img from the point(x1,y1) to (x2,y2). (0,0,255) denotes the colour of the line to be drawn. In this case, it is red.
        cv2.line(marked_img, (x1, y1), (x2, y2), marked_color, 2)
        lines_list.append([(x1, y1), (x2, y2)])

    # All the changes made in the input image are finally written on a new image houghlines.jpg
    if showResult == True:
        cv2.imshow('linesDetected.jpg', marked_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


    return marked_img, lines_list


#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
def F1(img):
    dest = Harris_Corner_Detector(img,showResult=True)
def F2(img):
    min_factor = min(img.shape[0]//128, img.shape[1]//64)
    hog_img = HOG_transform(img,resize=(128*min_factor,64*min_factor))
    cv2.imshow('original image', img)
    cv2.imshow('HOG image',hog_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def F3(img):
    new_img = Canny_Edge_Detector(img,params=(50,150))
    cv2.imshow('original image', img)
    cv2.imshow('Canny image', new_img)
    cv2.waitKey()
    cv2.destroyAllWindows()
def F4(img):
    Hough_Transform_easy(img,showResult=True)
    Hough_Transform_hard(img,showResult=True)

def printMenu():
    print('*'*20,' MENU ','*'*20)
    print('1. Function 1: Harris Corner Detector ')
    print('2. Function 2: The Histogram of Oriented Gradients ')
    print('3. Function 3: Canny Operator ')
    print('4. Function 4: Hough transform ')
    print('5. EXIT.')


if __name__ == '__main__':
    import cv2
    ifol = r'K:\LEARN AI FPT\PROJECTS\images\\'
    img = cv2.imread(ifol + 'tgb.png')

    while True:
        printMenu()
        opt = int(input('Choose [1..5]: '))
        if opt < 1 or opt >= 5:
            quit()
        elif opt == 1:
            F1(img)
        elif opt == 2:
            F2(img)
        elif opt == 3:
            F3(img)
        else:
            F4(img)
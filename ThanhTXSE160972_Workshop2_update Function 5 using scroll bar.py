import cv2
import numpy as np
import matplotlib.pyplot as plt

def compute_hist(img):
    hist = [0] * 256
    hist = np.array(hist)
    (h,w) = img.shape[:2]
    for i in range(h):
        for j in range(w):
            light_intensity = img[i][j]
            hist[light_intensity] += 1
    return hist

def equalizeHist(hist):
    cum = np.zeros_like(hist)
    for i in range(len(cum)):
        cum[i] = hist[:i].sum()
    # scale cum's elements:
    cum = (cum - cum.min())/(cum.max() - cum.min()) * 255
    cum = [int(i) for i in cum]
    cum = np.array(cum)
    return cum

def brightness_colored(img):
    hist = compute_hist(img)
    cum = equalizeHist(hist)
    new_img = np.zeros_like(img)
    (h, w) = new_img.shape[:2]
    for i in range(h):
        for j in range(w):
            new_img[i][j] = cum[img[i][j]]
    return new_img

def color_balanced(val,index,image):
    global img_cb
    val = int(val*2.55)
    img_cb[:,:,index] = image[:, :, index] + val
    cv2.imshow("Color balance", img_cb)

def F1(img):
    global img_cb
    img_cb = img.copy()
    cv2.imshow('Original image',img)
    cv2.namedWindow("Color balance", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Blue", "Color balance", 0, 100, lambda val: color_balanced(val=val, index=0, image=img))
    cv2.createTrackbar("Green", "Color balance", 0, 100, lambda val: color_balanced(val=val, index=1, image=img))
    cv2.createTrackbar("Red", "Color balance", 0, 100, lambda val: color_balanced(val=val, index=2, image=img))
    cv2.waitKey()
    cv2.destroyAllWindows()


def F2(img):
    new_img = np.zeros_like(img)
    for i in range(3):
        new_img[:,:,i] = brightness_colored(img[:,:,i])
    old_hist = np.zeros((3,256))
    new_hist = np.zeros_like(old_hist)
    for i in range(3):
        old_hist[i] = compute_hist(img[:,:,i])
        new_hist[i] = compute_hist(new_img[:,:,i])

    cv2.imshow('original image', img)
    cv2.imshow('histogram equalized image', new_img)

    figure,axis = plt.subplots(1,2,figsize=(12,8))

    colors = ['blue','green','red']
    for i in range(3):
        axis[0].plot(range(256),old_hist[i],color=colors[i][0],label=colors[i])
        axis[1].plot(range(256), new_hist[i], color=colors[i][0], label=colors[i])

    axis[0].legend(loc='best')
    axis[1].legend(loc='best')
    axis[0].set(xlabel = 'Value', ylabel = 'Frequency', title = 'Before histogram equalization')
    axis[1].set(xlabel='Value', ylabel='Frequency', title='After histogram equalization')

    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()


def MeanMedian_filter(img,kind,type = 0, kernel_size = 3):
    if kernel_size % 2 == 0:
        print('ERROR: kernel_size cannot be even.')
        return img
    if kind != 'Mean' and kind != 'Median':
        print("ERROR: kind must be 'Mean' or 'Median'.")
        return img

    new_img = np.zeros_like(img)
    h, w = img.shape[:2]
    center = kernel_size // 2
    kernel = np.zeros((kernel_size,kernel_size))

    if type == 0:  # ----- biên ảnh mới = biên ảnh cũ: DONE!
        for i in range(center):
            new_img[i] = img[i]
            new_img[-i-1] = img[-i-1]
        for i in range(center, h - center):
            for j in range(center):
                new_img[i,j] = img[i][j]
                new_img[i,-j-1] = img[i,-j-1]
        #--- kernel applied:
        for i in range(h - kernel_size + 1):
            for j in range(w - kernel_size + 1):
                for k in range(3):
                    kernel = img[i:i + kernel_size, j:j + kernel_size, k]

                    if kind == 'Mean':
                        v = np.mean(kernel)
                    elif kind == 'Median':
                        v = np.median(kernel)

                    new_img[i + center, j + center, k] = int(v)

    elif type == 1:  # -------- extend biên with chính nó: DONE !
        img_extended = np.zeros((h+center*2,w+center*2,3))
        for i in range(center): #--- fill top and bottom rows
            for j in range(center,w+center):
                img_extended[i, j] = img[0, j - center]
                img_extended[-i-1, j] = img[-1, j - center]
        for i in range(center, h + center): #--- fill left and right columns
            for j in range(center):
                img_extended[i,j] = img[i - center, 0]
                img_extended[i,-j-1] = img[i - center, -1]
        for i in range(center): #--- fill pixel at corners with corner.
            for j in range(center):
                img_extended[i,j] = img[0, 0]
                img_extended[i, -j - 1] = img[0, -1]
                img_extended[-i - 1, j] = img[-1][0]
                img_extended[-i - 1, -j - 1] = img[-1][-1]
        for k in range(3):  # --- fill centre with original image.
            img_extended[center:h + center, center:w + center, k] = img[:,:,k].copy()
        #img_extended[center:h + center, center:w + center] = img.copy()  # --- fill centre with original image.
        #--------------- TEST img_extended --------------
        '''cv2.imshow('1',img)
        cv2.imshow('2',img_extended)
        cv2.waitKey()
        cv2.destroyAllWindows()'''
        #--- kernel applied:
        h, w = img_extended.shape[:2]  #--- NOTE THAT: h now is of img_extended
        for i in range(h - kernel_size + 1):
            for j in range(w - kernel_size + 1):
                for k in range(3):
                    kernel = img_extended[i:i + kernel_size, j:j + kernel_size, k]

                    if kind == 'Mean':
                        v = np.mean(kernel)
                    elif kind == 'Median':
                        v = np.median(kernel)

                    new_img[i, j, k] = int(v)

    elif type == 2:  # --------- extend biên with 0s: DONE !
        img_extended = np.zeros((h + center * 2, w + center * 2, 3))
        for k in range(3):  # --- fill centre with original image.
            img_extended[center:h + center, center:w + center, k] = img[:,:,k].copy()
        #--------------- TEST img_extended --------------
        '''cv2.imshow('1',img)
        cv2.imshow('2',img_extended)
        cv2.waitKey()
        cv2.destroyAllWindows()'''
        #--- kernel applied:
        h, w = img_extended.shape[:2]  #--- NOTE THAT: h now is of img_extended
        for i in range(h - kernel_size + 1):
            for j in range(w - kernel_size + 1):
                for k in range(3):
                    kernel = img_extended[i:i + kernel_size, j:j + kernel_size, k]

                    if kind == 'Mean':
                        v = np.mean(kernel)
                    elif kind == 'Median':
                        v = np.median(kernel)

                    new_img[i, j, k] = int(v)


    return new_img

def F4(img,type = 0, kernel_size = 3):
    new_img = MeanMedian_filter(img=img,kind='Mean',type=type,kernel_size=kernel_size)
    cv2.imshow('original image', img)
    cv2.imshow('Mean filter preview', new_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def F3(img,type = 0, kernel_size = 3):
    new_img = MeanMedian_filter(img=img,kind='Median',type=type,kernel_size=kernel_size)
    cv2.imshow('original image', img)
    cv2.imshow('Median filter preview', new_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def F5(img,kernel_size=3,sigma=1.5):
    # Lọc ảnh bằng Gaussian filter
    filtered_img = gaussian_filter(img, kernel_size, sigma)
    # Show kết quả:
    cv2.imshow('Original image', img)
    cv2.imshow('Gaussian filter preview', filtered_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def gaussian_filter(img, kernel_size=3, sigma=1.5):
    # Tạo ma trận kernel Gaussian
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    # Tích chập với mỗi vùng của ảnh đầu vào
    filtered_img = np.zeros_like(img)
    for i in range(img.shape[0] - kernel_size + 1):
        for j in range(img.shape[1] - kernel_size + 1):
            for k in range(3):
                v = np.sum(img[i:i + kernel_size, j:j + kernel_size, k] * kernel)
                filtered_img[i + center, j + center, k] = v

    return filtered_img

def F5_v2(img):
    global img_GF
    img_GF = img.copy()
    cv2.imshow('Original image', img)
    GW = np.zeros((256,750,3))
    GW.fill(255)
    cv2.imshow("GAUSSIAN filter", GW)
    cv2.createTrackbar("Sigma", "GAUSSIAN filter", 1, 50, lambda val: gaussian_filter_v2(img=img, sigma=val/4))
    cv2.createTrackbar("Kernel size", "GAUSSIAN filter", 0, 50,lambda val: gaussian_filter_v2(img=img, kernel_size=val*2+1))

    cv2.waitKey()
    cv2.destroyAllWindows()

def gaussian_filter_v2(img, kernel_size=3, sigma=1.5):
    # Tạo ma trận kernel Gaussian
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)

    # Tích chập với mỗi vùng của ảnh đầu vào
    filtered_img = np.zeros_like(img)
    for i in range(img.shape[0] - kernel_size + 1):
        for j in range(img.shape[1] - kernel_size + 1):
            for k in range(3):
                v = np.sum(img[i:i + kernel_size, j:j + kernel_size, k] * kernel)
                filtered_img[i + center, j + center, k] = v

    cv2.imshow('GAUSSIAN Preview',filtered_img)

    return filtered_img

def printMenu():
    print('-' * 20 + ' MENU ' + '-' * 20)
    print('Function 1: color balance.')
    print('Function 2: Show histogram and perform histogram equalization.')
    print('Function 3: implement the Median filter to remove noise in the image (salt and pepper noise)')
    print('Function 4: implement the Mean filter to remove noise in image (salt and pepper noise)')
    print('Function 5: implement Gaussian smoothing to perform image smoothing.')
    print('6. EXIT\n')
# ------------------------------------------------------------------------
# ------------------------------------------------------------------------

if __name__ == '__main__':
    folder = r'K:\LEARN AI FPT\PROJECTS\PycharmProjects\images\\'
    img_name = input('Enter image name: ')
    if len(img_name) < 1:
        img_name = 'lena_saltpepper.png'
    img = cv2.imread(folder + img_name)

    while True:
        printMenu()
        opt = int(input('Choose [1..6]: '))
        if opt < 1 or opt >= 6:
            print('Good bye!. ')
            quit()
        elif opt == 1:
            F1(img)
        elif opt == 2:
            F2(img)
        elif opt == 3:
            print('type 0: keep borders as original')
            print('type 1: extend borders with border before filter')
            print('type 2: extend borders with 0s before filter')
            typ = int(input('Choose type [0..2]: '))
            ker_size = int(input('Enter kernel size (e.g: 3): '))
            F3(img,type=typ,kernel_size=ker_size)
        elif opt == 4:
            print('type 0: keep borders as original')
            print('type 1: extend borders with border before filter')
            print('type 2: extend borders with 0s before filter')
            typ = int(input('Choose type [0..2]: '))
            ker_size = int(input('Enter kernel size (e.g: 3): '))
            F4(img,type=typ,kernel_size=ker_size)
        elif opt == 5:
            #sigma = float(input('Enter sigma value (e.g: 1.5): '))
            #ker_size = int(input('Enter kernel size (e.g: 3): '))
            F5_v2(img)


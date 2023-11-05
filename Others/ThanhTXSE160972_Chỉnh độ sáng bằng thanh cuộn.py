import cv2
import numpy as np

#-------------- KHAI BÁO CÁC BIẾN GLOBAL -----------------
folder = r'K:\LEARN AI FPT\PROJECTS\PycharmProjects\images\\'
image_name = 'sd.jpg'
get_position = False
thickness = 1
blue = (255,0,0)
white = (255,255,255)
green = (0,255,0)
red = (0,0,255)
cursor_size = 10
A = (75,70)
C = (125,580)
B = (C[0],A[1])
D = (A[0],C[1])
M = ( (A[0]+B[0])//2,(A[1]+B[1])//2 )
N = ( (C[0]+D[0])//2,(C[1]+D[1])//2 )
I = ( (A[0]+C[0])//2,(A[1]+C[1])//2 )
mode = False
moving = False
value = None
value_change = False
lst = [I[0],I[1]]
lim = 15                   #----------- 2 sides lim of I.
I_lim_left, I_lim_right = (I[0] - lim, I[1]) , (I[0] + lim, I[1])
#----------
#-------------------------------
def brightness_change(img,value):
    main_image = img
    h, w = main_image.shape[:2]
    if str(value)[0] == '+':
        value = int(str(value)[1:])
        value = value // 2
        main_image = main_image + value
    elif str(value)[0] == '-':
        value = int(str(value)[1:])
        value = value // 2
        main_image = main_image - value
    return main_image
# -------------------------------
def drawRectangle(top_left,right_bottom,color = (0,0,255),thickness=1):
    global window, red, blue
    A = top_left
    C = right_bottom
    B = (C[0], A[1])
    D = (A[0], C[1])
    color = blue
    window = cv2.line(window, A, B, color, thickness)
    window = cv2.line(window, B, C, color, thickness)
    window = cv2.line(window, C, D, color, thickness)
    window = cv2.line(window, D, A, color, thickness)
def recover_frame():
    global window,A,B,C,D,M,N,I,I_lim_left,I_lim_right,red,blue,thickness
    drawRectangle(A,C)
    window = cv2.line(window, M, N, red, thickness)
    window = cv2.line(window, I_lim_left, I_lim_right, red, 5)
#----------------------
#------------------------
def showValue(value):
    labelPoint(x=I[0] + 40, y=I[1] + 10, image=window, name=str(value), thickness_point=0,
               color_point=red, color_text=red,display_coors=False)
def deleteValue():
    global window, green,white
    top_left = ( I[0] + 40, I[1] - 20 )
    right_bottom = ( I[0] + 100, I[1] + 10 )
    window = cv2.rectangle(window,top_left,right_bottom,white,-1)


#-------------------------
def getValues(event,x,y,flag,params):
    global mode,window,lst,draw, moving, value, get_position, value_change
    if event == cv2.EVENT_LBUTTONDOWN:
        if get_position == True:
            labelPoint(x,y,window,name='')
        if x > A[0] and x < C[0] and y > A[1] and y < C[1]:
            # --- delete old cursor and recover the centre line:
            window = cv2.circle(window, (I[0], lst[1]), cursor_size, white, -1)
            window = cv2.circle(window, N, cursor_size, white, -1)
            window = cv2.circle(window, M, cursor_size, white, -1)
            recover_frame()
            # ---
            mode = True
            lst = [x,y]
            window = cv2.circle(window, (I[0], lst[1]), cursor_size, red, -1)
            value_change = True
            # --- show value:
            deleteValue()
            value = I[1] - lst[1]
            if value > 0: value = '+' + str(value)
            showValue(value)

    elif event == cv2.EVENT_MOUSEMOVE:
        if mode == True and \
            x > A[0] and x < C[0] and y > A[1] and y < C[1] :
            value = I[1] - y
            # --- delete old cursor and recover the centre line:
            window = cv2.circle(window, (I[0],lst[1]), cursor_size, white, -1)
            window = cv2.circle(window, N, cursor_size, white, -1)
            window = cv2.circle(window, M, cursor_size, white, -1)
            recover_frame()
            # ---
            cursor = (x,y)
            window = cv2.circle(window, (I[0], cursor[1]), cursor_size, red, -1)
            value_change = True
            # --- show value:
            deleteValue()
            value = I[1] - cursor[1]
            if value > 0: value = '+' + str(value)
            showValue(value)
            # ---
            lst = [x,y]
            moving = True
    elif event == cv2.EVENT_LBUTTONUP:
        if mode == True:
            # --- delete old cursor and recover the centre line:
            window = cv2.circle(window, (I[0], lst[1]), cursor_size, white, -1)
            window = cv2.circle(window, N, cursor_size, white, -1)
            window = cv2.circle(window, M, cursor_size, white, -1)
            recover_frame()
            # ---
            cursor = (x, y)
            if x > A[0] and x < C[0] and y > A[1] and y < C[1]:
                window = cv2.circle(window, (I[0], lst[1]), cursor_size, red, -1)
                # --- show value:
                deleteValue()
                deleteValue()
                value = I[1] - lst[1]
                if value > 0: value = '+' + str(value)
                showValue(value)
                # ---
            elif y >= C[1]:
                window = cv2.circle(window, N, cursor_size, red, -1)
                # --- show value:
                deleteValue()
                value = I[1] - N[1]
                if value > 0: value = '+' + str(value)
                showValue(value)
                # ---
            elif y <= A[1]:
                window = cv2.circle(window, M, cursor_size, red, -1)
                # --- show value:
                deleteValue()
                value = I[1] - M[1]
                if value > 0: value = '+' + str(value)
                showValue(value)
                # ---
            elif moving == True:
                window = cv2.circle(window, (I[0], y), cursor_size, red, -1)
                # --- show value:
                deleteValue()
                value = I[1] - y
                if value > 0: value = '+' + str(value)
                showValue(value)
                # ---

            value_change = True
            lst = [x, y]

            mode = False

    cv2.imshow('brightness',window)

    if value is not None and value_change == True:
        main_image_processing(value)
        value_change = False

#-------------------------------------
def main_image_processing(value):
    global main_image, window

    new_image = brightness_change(main_image,value)
    #new_image = main_image + value/255*(255-main_image)
    window = cv2.rectangle(window, (0, 0), (87, 38), white, -1)
    if (main_image == new_image).all():
        labelPoint(20,20,image=window,name='True',display_coors=False)
    else:
        labelPoint(20, 20, image=window, name='False', display_coors=False)

    cv2.imshow('image preview', new_image)


#-------------------------------------
def labelPoint(x, y, image, name='A', thickness_point=3, color_point=(0,0,255), color_text=(0,0,255),
               display_coors = True):
    cv2.circle(image, (x, y), thickness_point, color_point, -1)
    if display_coors == True:
        strXY = str(name) + ' (' + str(x) + ',' + str(y) + ')'
    else:
        strXY = str(name)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, strXY, (x + 10, y - 10), font, 1, color_text)


#--------------- IMAGE WINDOW --------------------
main_image = cv2.imread(folder + image_name)
#---
cv2.imshow('original image', main_image)
#--------------- BRIGHTNESS WINDOW ----------------
window = np.zeros((670,200,3), dtype=np.uint8)
window.fill(255)
#--------------------------
window = cv2.circle(window, I, cursor_size, red, -1)
recover_frame()
#-------------------------
cv2.imshow('brightness',window)
#---------------------------------------------------

cv2.setMouseCallback('brightness',getValues)


cv2.waitKey()
cv2.destroyAllWindows()
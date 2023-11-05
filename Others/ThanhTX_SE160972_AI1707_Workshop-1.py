import cv2
import numpy as np

global color, thickness, mode
color = (255, 0, 0)  # --- B-G-R
thickness = 2
draw = False
mode = False
lst = []


def labelPoint(x, y, image, name='A', thickness_point=3, color_point=color, color_text=color):
    cv2.circle(image, (x, y), thickness_point, color_point, -1)
    strXY = str(name) + ' (' + str(x) + ',' + str(y) + ')'
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.putText(image, strXY, (x + 10, y - 10), font, 1, color_text)


def getCoordinate(pointName):
    return pointName[0], pointName[1]


def drawRectangle(pointA, pointC, image, thickness_line=thickness, color_line=color):
    pointD = (pointA[0], pointC[1])
    pointB = (pointC[0], pointA[1])
    print('pointA: ', pointA)
    print('pointB: ', pointB)
    print('pointC: ', pointC)
    print('pointD: ', pointD)
    points = [pointA, pointB, pointC, pointD]
    pointNames = ['A', 'B', 'C', 'D']
    for i in range(len(points)):
        x, y = getCoordinate(points[i])
        labelPoint(x, y, image, pointNames[i], color_point=color_line, color_text=color_line)
    cv2.line(image, pointA, pointB, color_line, thickness_line)
    cv2.line(image, pointB, pointC, color_line, thickness_line)
    cv2.line(image, pointC, pointD, color_line, thickness_line)
    cv2.line(image, pointD, pointA, color_line, thickness_line)


def dragRectangle(event, x, y, flag, params):
    global screen_name, lst
    global draw, p1, p2, color, thickness, img, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        p1 = (x, y)
        labelPoint(x, y, img)
        lst = [p1]
        cv2.imshow(screen_name, img)
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            drawRectangle(lst[0], lst[-1], img, thickness, (255, 255, 255))
            p2 = (x, y)
            drawRectangle(p1, p2, img, thickness, color)
            lst = [p1, p2]
            mode = True
            cv2.imshow(screen_name, img)
    elif (event == cv2.EVENT_LBUTTONUP):
        draw = False
        lst = []
        if mode == True:
            p2 = (x, y)
            drawRectangle(p1, p2, img, thickness, color)
            # hàm có sẵn: img = cv2.rectangle(img, p1, p2, color, thickness)
            mode = False
            cv2.imshow(screen_name, img)
        else:
            labelPoint(x, y, img)


def printMenu():
    print('-' * 20 + ' MENU ' + '-' * 20)
    print('1. Create a white background')
    print('2. Drag to draw a rectangle')
    print('3. Transition transformation (Using mouse).')
    print('4. Enter rotation angle information')
    print('5. Scaling transformation')
    print('6. EXIT')


def transitionRectangle(event, x, y, flag, params):
    global screen_name
    global draw, start, end, color, thickness, img, mode, T
    # -----------
    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True
        start = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if draw == True:
            mode = True
    elif (event == cv2.EVENT_LBUTTONUP):
        draw = False
        if mode == True:
            end = (x, y)
            T = (end[0] - start[0], end[1] - start[1])
            p1_new = (p1[0] + T[0], p1[1] + T[1])
            p2_new = (p2[0] + T[0], p2[1] + T[1])
            new_color = (0, 0, 255)
            drawRectangle(p1_new, p2_new, img, thickness, new_color)
            cv2.imshow(screen_name, img)
            mode = False
        else:
            labelPoint(x, y, img)


def scaleRec(event, x, y, flag, params):
    if (event == cv2.EVENT_LBUTTONDOWN):
        centre = [x, y]
        new_color = (0, 0, 255)
        pointA = p1
        pointC = p2
        pointD = (pointA[0], pointC[1])
        pointB = (pointC[0], pointA[1])
        points = [pointA, pointB, pointC, pointD]
        points_new = [0] * len(points)
        for i in range(len(points)):
            pt_new = [int(points[i][0] - centre[0]), int(points[i][1] - centre[1])]
            xnew = int(pt_new[0] * S_x)
            ynew = int(pt_new[1] * S_y)
            pt_new_rot = [xnew, ynew]
            points_new[i] = (pt_new_rot[0] + centre[0], pt_new_rot[1] + centre[1])
        pointNames = ["A'", "B'", "C'", "D'"]

        labelPoint(centre[0], centre[1], img, 'I ' + str(S_x) + ':' + str(S_y), 5, new_color, new_color)
        for i in [-1, 0, 1, 2]:
            labelPoint(points_new[i + 1][0], points_new[i + 1][1], img, pointNames[i + 1], 3, new_color, new_color)
            cv2.line(img, points_new[i], points_new[i + 1], new_color, thickness)
        cv2.imshow(screen_name, img)


def rotateRec(event, x, y, flag, params):
    global angle
    if (event == cv2.EVENT_LBUTTONDOWN):
        centre = [x, y]
        new_color = (0, 0, 255)
        pointA = p1
        pointC = p2
        pointD = (pointA[0], pointC[1])
        pointB = (pointC[0], pointA[1])
        points = [pointA, pointB, pointC, pointD]
        points_new = [0] * len(points)
        for i in range(len(points)):
            pt_new = [int(points[i][0] - centre[0]), int(points[i][1] - centre[1])]
            xnew = int(pt_new[0] * np.cos(angle) - pt_new[1] * np.sin(angle))
            ynew = int(pt_new[0] * np.sin(angle) + pt_new[1] * np.cos(angle))
            pt_new_rot = [xnew, ynew]
            points_new[i] = (pt_new_rot[0] + centre[0], pt_new_rot[1] + centre[1])
        pointNames = ["A'", "B'", "C'", "D'"]

        labelPoint(centre[0], centre[1], img, 'I', 5, new_color, new_color)
        for i in [-1, 0, 1, 2]:
            labelPoint(points_new[i + 1][0], points_new[i + 1][1], img, pointNames[i + 1], 3, new_color, new_color)
            cv2.line(img, points_new[i], points_new[i + 1], new_color, thickness)
        cv2.imshow(screen_name, img)


# ------------------------------------------------
if __name__ == '__main__':
    screen_name = 'Main'
    while True:
        printMenu()
        opt = int(input())
        if opt < 1 or opt >= 6:
            quit()
        else:
            if opt == 1:
                img = np.zeros([512, 1024, 3], dtype=np.uint8)
                img.fill(255)
                cv2.imshow(screen_name, img)
                cv2.waitKey()
                cv2.destroyAllWindows()
                continue
            elif opt == 2 or opt == 3:
                img = np.zeros([512, 1024, 3], dtype=np.uint8)
                img.fill(255)
                cv2.imshow(screen_name, img)
                # --- drag Rectangle ---
                cv2.setMouseCallback(screen_name, dragRectangle)
                cv2.waitKey()
                # --- transition ---
                cv2.setMouseCallback(screen_name, transitionRectangle)
                cv2.waitKey()
                cv2.destroyAllWindows()
            elif opt == 4:
                global angle
                angle = float(input('Enter rotation angle (degree): '))
                angle = -angle / 180 * np.pi
                # --- drag Rectangle ---
                img = np.zeros([512, 1024, 3], dtype=np.uint8)
                img.fill(255)
                cv2.imshow(screen_name, img)
                cv2.setMouseCallback(screen_name, dragRectangle)
                cv2.waitKey()
                # --- scaling ---
                cv2.setMouseCallback(screen_name, rotateRec)
                cv2.waitKey()
                cv2.destroyAllWindows()
            elif opt == 5:
                S = input('Enter scaling factors (for e.g: 2 2.5): ')
                S = [float(k) for k in S.strip().split(' ')]
                S_x, S_y = S[0], S[1]
                # --- drag Rectangle ---
                img = np.zeros([512, 1024, 3], dtype=np.uint8)
                img.fill(255)
                cv2.imshow(screen_name, img)
                cv2.setMouseCallback(screen_name, dragRectangle)
                cv2.waitKey()
                # --- scaling ---
                cv2.setMouseCallback(screen_name, scaleRec)
                cv2.waitKey()
                cv2.destroyAllWindows()
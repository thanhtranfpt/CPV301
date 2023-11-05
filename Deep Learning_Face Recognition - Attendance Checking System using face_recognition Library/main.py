'''
    Tran Xuan Thanh - FPT University HCM.
    facebook.com/TXTPTTB
'''


import os
from datetime import datetime

import face_recognition
import face_recognition_models
import cv2
import numpy as np

from General_Functions import write_to_csv
from General_Functions import Text_to_Speech
from General_Functions import play_sound
from General_Functions import Unicode_to_ASCII

from FilterImage import brightness_colored

#-----------------------------------------------------------------
def encodeCamFrame(img,draw_rect = False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encode_img = face_recognition.face_encodings(img)[0]
    faceLoc = face_recognition.face_locations(img)[0]
    if draw_rect == True:
        cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (0, 255, 0), 2)
        cv2.imshow('Image Detection', img)
        cv2.waitKey()
    return encode_img,faceLoc

def encodeImage(ipath,draw_rect = False):
    img = face_recognition.load_image_file(ipath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    encode_img = face_recognition.face_encodings(img)[0]
    faceLoc = face_recognition.face_locations(img)[0]
    if draw_rect == True:
        cv2.rectangle(img,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(0,255,0),2)
        cv2.imshow('Image Detection',img)
        cv2.waitKey()
    return encode_img,faceLoc

def compare_faces(encode_img1,encode_img2):
    return face_recognition.compare_faces([encode_img1],encode_img2)[0]
def images_distance(encode_img1,encode_img2):
    return face_recognition.face_distance([encode_img1],encode_img2)[0]
#-----------------------------------------------------------------
#-----------------------------------------------------------------

def get_StudentsData(file_path='File danh sách lớp.csv'):
    import pandas as pd
    import numpy as np

    df = pd.read_csv(file_path)
    df['Full_Name'] = df['CODE'] + ' ' + df['SURNAME'] + ' ' + df['MIDDLE NAME']
    df.set_index('MEMBER', inplace=True)
    df['Check_Attendance'] = 'Not yet'
    df['Time_Attendance'] = ''
    # df.loc['SE160972','Check_Attendance'] = 'Attended'
    # fullName = df.loc['SE160972','Full_Name']
    # 'me' in df.index

    return df

def get_lstEncode(ifolder):
    lstNames = []
    for file_name in os.listdir(ifolder):
        lstNames.append(file_name)

    lstEncode = []
    for file_name in lstNames:
        ipath = ifolder + file_name
        encode_img = encodeImage(ipath)[0]
        lstEncode.append(encode_img)

    return lstNames, lstEncode


def takeAttendance(row):
    global df
    print(row)
    write_to_csv(r'K:\LEARN AI FPT\PROJECTS\Final_Project_CPV301_Hệ thống điểm danh khuôn mặt\\File điểm danh.csv', row)
    play_sound(r"K:\LEARN AI FPT\PROJECTS\sounds\tick2.wav")
    Text_to_Speech(row[1] + ' ' + row[0] + ' ' + row[-1])
    df.loc[row[0], 'Check_Attendance'] = 'Attended'
    df.loc[row[0], 'Time_Attendance'] = row[-2] + ' ' + row[-1]
    df.to_csv('File điểm danh FULL.csv', encoding='UTF-8')


def PreProcessImage(img):
    return brightness_colored(img)

#-----------------------------------------------------------------
ifolder = r'K:\LEARN AI FPT\PROJECTS\Final_Project_CPV301_Hệ thống điểm danh khuôn mặt\images\\'

df = get_StudentsData('File danh sách lớp.csv')
lstNames, lstEncode = get_lstEncode(ifolder)

def getNamefromID(id):
    def get_StudentsData(file_path='File danh sách lớp.csv'):
        import pandas as pd
        import numpy as np

        df = pd.read_csv(file_path)
        df['Full_Name'] = df['CODE'] + ' ' + df['SURNAME'] + ' ' + df['MIDDLE NAME']
        df.set_index('MEMBER', inplace=True)
        df['Check_Attendance'] = 'Not yet'
        df['Time_Attendance'] = ''
        # df.loc['SE160972','Check_Attendance'] = 'Attended'
        # fullName = df.loc['SE160972','Full_Name']
        # 'me' in df.index

        return df
    df = get_StudentsData('File danh sách lớp.csv')
    name = df.loc[id,'Full_Name']

    return name

# ===== ======
scale = 4
scaledown = 1/scale

# ===== CAMERA ======

cap = cv2.VideoCapture(0)

while True:
    #--- ghi vào tệp
    data = []
    #---
    success, img = cap.read()

    # --- XỬ LÝ ẢNH_KHUNG HÌNH CHO RÕ, NÉT: img
    # img = PreProcessImage(img)
    # --- END
    imgS = cv2.resize(img, (0, 0), None, scaledown, scaledown)
    # imgS = img

    face_Locs = face_recognition.face_locations(imgS)  # --- returns face's locations in imgS
    encodes = face_recognition.face_encodings(imgS)  # --- returns encode_img of each face location in imgS.

    for faceLoc, encode_img in zip(face_Locs, encodes):
        faceLoc = [q * scale for q in faceLoc]
        top, right, bottom, left = faceLoc

        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.rectangle(img, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)

        # calculate distances to every known faces:
        distances = face_recognition.face_distance(lstEncode, encode_img)

        min_pos = np.argmin(distances)
        if compare_faces(lstEncode[min_pos], encode_img) == True:
            sid = lstNames[min_pos][:-4]
            if sid in df.index:
                fullName = df.loc[sid, 'Full_Name']
                name_showed = fullName.split(' ')[-2] + ' ' + fullName.split(' ')[-1]
            else:
                name_showed, fullName = [sid] * 2
        else:
            sid, name_showed, fullName = ['Unknown!'] * 3

        cv2.putText(img, Unicode_to_ASCII(name_showed), (left + 6, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

        # ---- GET DATE TIME
        now = datetime.now()
        now = str(now.strftime("%d/%m/%Y %H:%M:%S"))
        cv2.putText(img, now.split(' ')[1], (left + 26, top - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        data.append([sid, fullName, now.split(' ')[0], now.split(' ')[1]])

        # --- END

    cv2.imshow('Look at me!', img)

    key = cv2.waitKey(10)

    if key == 13:  # --- Enter
        for row in data:
            takeAttendance(row)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
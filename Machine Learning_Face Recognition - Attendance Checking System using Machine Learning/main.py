'''
    Tran Xuan Thanh _ FPT University HCM.
    facebook.com/TXTPTTB
'''

import os
import cv2
import numpy as np
from General_Functions import Unicode_to_ASCII
from General_Functions import Text_to_Speech
from General_Functions import write_to_csv, play_sound
from datetime import datetime
from FilterImage import brightness_colored, brightness_colord_by_cv2, medianBlur_by_cv2


def get_faces_from_folder(image_folder_path = r'...'):
    '''
    input: folder path containing images that needs to be cropped faces.
    output: list of faces and its corresponding labels (based on its file name).
    '''
    from General_Functions import HaarCascade_face_detection
    from General_Functions import getMaxFacefromImage
    import os
    import cv2
    import numpy as np

    faces = []
    labels = []
    c, s = 0, len(os.listdir(image_folder_path))
    for file_name in os.listdir(image_folder_path):
        impath = image_folder_path + r'\\' + file_name
        img = cv2.imread(impath)
        img, _ = getMaxFacefromImage(img,myResize=(32,32))

        if _ is None:
            img = cv2.resize(img,(32,32))
            print('WARNING: Cannot find any face in the image : ', file_name, ' !')

        faces.append(img)
        labels.append(file_name.strip().split('.')[0])      #assume that the file name is: person1.2.jpg

    return faces, labels


def over_sample_face_images(image_list, label_list, output_folder = r'...\\'):
    '''
    input: list of face-images and its corresponding labels.
    output: new list of face-images and its corresponding labels.
    '''
    import cv2_ext
    import numpy as np
    import cv2
    import os

    # define the list of labels and faces
    faces, labels = image_list, label_list
    faces = [cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) for face in faces]
    faces = np.array(faces)
    print('Old sample size : ',len(faces))

    # define the oversampling parameters
    num_samples = 10
    flip_prob = 0.5
    max_angle = 30
    max_scale = 0.2
    target_size = (32,32)  # define a fixed target size for the images: (32, 32)

    # define the list of augmented faces and labels
    aug_faces = []
    aug_labels = []

    # perform oversampling by flipping, rotating, and cropping the faces
    for i in range(len(faces)):
        for j in range(num_samples):
            # randomly flip the image horizontally
            if np.random.rand() < flip_prob:
                img = cv2.flip(np.array(faces[i]), 1)
            else:
                img = faces[i]

            # randomly rotate the image
            angle = np.random.randint(-max_angle, max_angle)
            scale = 1 + np.random.uniform(-max_scale, max_scale)
            M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, scale)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

            # randomly crop the image
            crop_size = np.random.randint(int(img.shape[0] * 0.8), img.shape[0])
            x = np.random.randint(0, img.shape[1] - crop_size)
            y = np.random.randint(0, img.shape[0] - crop_size)
            img = img[y:y + crop_size, x:x + crop_size]

            # resize the image to the target size
            img = cv2.resize(img, target_size)

            # add the augmented face to the list
            aug_faces.append(img)
            aug_labels.append(labels[i])

    # resize all the images in the original faces list to the target size
    for i in range(len(faces)):
        faces[i] = cv2.resize(faces[i], target_size)

    # concatenate the original and augmented faces and labels
    faces = np.concatenate([faces, np.array(aug_faces)], axis=0)
    labels = labels + aug_labels

    print('New sample size : ',len(faces))

    if output_folder is not None:
        for i, face in enumerate(faces):
            cv2_ext.imwrite(output_folder + labels[i] + '.' + str(i) + '.png', face)
            print('Writing ' + labels[i] + '.' + str(i) + '.png successfully!')

    return faces, labels

def encodeFace(face):
    import cv2
    import numpy as np
    from General_Functions import HOG_transform

    dst = cv2.GaussianBlur(face, (5, 5), 0)
    _, fd = HOG_transform(dst)
    fd = fd.flatten()

    return fd

def getData(image_folder_path = r"K:\LEARN AI FPT\PROJECTS\FinalCPVassignment_FaceRecognition_MachineLearning\over_training_images"):
    import numpy as np
    import cv2

    faces, labels = [], []

    for file_name in os.listdir(image_folder_path):
        impath = image_folder_path + r'\\' + file_name
        img = cv2.imread(impath)

        faces.append(img)
        labels.append(file_name.strip().split('.')[0])      #assume that the file name is: person1.2.jpg


    X = []
    for face in faces:
        encode_face = encodeFace(face)
        X.append(encode_face)

    X = np.array(X)
    y = labels

    return X,y

def buildKNNmodel(X,y):
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.005)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)

    print('Score KNN : ', score)
    '''for i, x in enumerate(X_test):
        print('True: ', y_test[i], ' - ', 'Predicted: ', knn.predict([x]))'''

    return knn, score

def buildSVMmodel(X,y):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    from sklearn.svm import SVC
    model = SVC(kernel='linear',C=1.0)
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)

    print('Score SVM : ',score)

    return model, score

def predictLabels(img,model):
    from General_Functions import HaarCascade_face_detection

    marked_img, face_locations, faces = HaarCascade_face_detection(img,showResult=False)

    X = []
    for face in faces:
        encode_face = encodeFace(face)
        X.append(encode_face)

    labels = []
    if len(X) > 0:
        labels = model.predict(X)

    return labels,face_locations

def showNames(img,labels,locations):
    for id,location in zip(labels,locations):
        name = getNamefromID(df,id)
        left, top = location[0]
        right,bottom = location[1]
        cv2.rectangle(img, location[0], location[1], (0, 255, 0), 2)
        cv2.rectangle(img, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
        #name_showed = name[-10:]
        cv2.putText(img, Unicode_to_ASCII(name), (left + 6, top - 6), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 255, 255), 2)

        # ---- GET DATE TIME
        now = datetime.now()
        now = str(now.strftime("%d/%m/%Y %H:%M:%S"))
        cv2.putText(img, now.split(' ')[1], (left + 26, top - 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        data.append([id, name, now.split(' ')[0], now.split(' ')[1]])

        # --- END

    return img

def getNamefromID(df,id):
    if id not in df.index:
        return "Unknown!"

    fullName = df.loc[id,'Full_Name']

    return fullName

def getStudentsData(file_path = r"K:\LEARN AI FPT\PROJECTS\Final_Project_CPV301_Hệ thống điểm danh khuôn mặt\File danh sách lớp.csv"):
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

def preProcess(img):
    img = brightness_colord_by_cv2(img)
    img = medianBlur_by_cv2(img)
    return img

def takeAttendance(row):
    global df
    print(row)
    write_to_csv(r"K:\LEARN AI FPT\PROJECTS\FinalCPVassignment_FaceRecognition_MachineLearning\\File_điểm_danh.csv", row)
    play_sound(r"K:\LEARN AI FPT\PROJECTS\sounds\tick2.wav")
    Text_to_Speech(row[1] + ' ' + row[0] + ' ' + row[-1])
    df.loc[row[0], 'Check_Attendance'] = 'Attended'
    df.loc[row[0], 'Time_Attendance'] = row[-2] + ' ' + row[-1]
    df.to_csv('File điểm danh FULL.csv', encoding='UTF-8')

#=============== === ===================
#----------------------------------------------------------------------
def EigenFacesAlgorithm(test_img,
                        train_image_folder_path = r"K:\LEARN AI FPT\PROJECTS\FinalCPVassignment_FaceRecognition_MachineLearning\images"):
    import cv2
    import os
    import numpy as np

    # Chuẩn bị dữ liệu
    dataset_path = train_image_folder_path
    face_images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            # label = int(filename.split(".")[0])
            label = str(filename.split(".")[0])
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_images.append(image)
            labels.append(label)

    # Tiền xử lý
    face_images = [cv2.resize(image, (100, 100)) for image in face_images]

    # Phân tích thành phần chính (PCA)
    X = np.array(face_images)
    X = X.reshape(X.shape[0], -1)
    mean, eigen_vectors = cv2.PCACompute(X, mean=None, maxComponents=10)

    # Xây dựng Eigenfaces
    eigen_faces = eigen_vectors.reshape(-1, 100, 100)

    # Huấn luyện mô hình
    face_space = np.dot(X - mean, eigen_vectors.T)
    face_labels = np.array(labels)

    # Nhận dạng khuôn mặt
    '''test_image = cv2.imread(r"K:\LEARN AI FPT\PROJECTS\images\AI1707_sampleFaces\numbersASI\ASI\6.1.jpg",
                            cv2.IMREAD_GRAYSCALE)'''
    test_image = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (100, 100))
    test_image_vector = test_image.reshape(1, -1)
    test_image_space = np.dot(test_image_vector - mean, eigen_vectors.T)
    distances = np.linalg.norm(face_space - test_image_space, axis=1)
    min_distance_index = np.argmin(distances)
    predicted_label = face_labels[min_distance_index]
    #print(predicted_label)

    return predicted_label

def EigenMethod_step1(train_image_folder_path=r"K:\LEARN AI FPT\PROJECTS\FinalCPVassignment_FaceRecognition_MachineLearning\images"):
    import cv2
    import os
    import numpy as np

    # Chuẩn bị dữ liệu
    dataset_path = train_image_folder_path
    face_images = []
    labels = []
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg"):
            # label = int(filename.split(".")[0])
            label = str(filename.split(".")[0])
            image_path = os.path.join(dataset_path, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            face_images.append(image)
            labels.append(label)

    # Tiền xử lý
    face_images = [cv2.resize(image, (100, 100)) for image in face_images]

    # Phân tích thành phần chính (PCA)
    X = np.array(face_images)
    X = X.reshape(X.shape[0], -1)
    mean, eigen_vectors = cv2.PCACompute(X, mean=None, maxComponents=10)

    # Xây dựng Eigenfaces
    eigen_faces = eigen_vectors.reshape(-1, 100, 100)

    # Huấn luyện mô hình
    face_space = np.dot(X - mean, eigen_vectors.T)
    face_labels = np.array(labels)

    return mean, eigen_vectors, face_space, face_labels

def EigenMethod_step2(test_img,mean, eigen_vectors, face_space, face_labels):
    test_image = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    test_image = cv2.resize(test_image, (100, 100))
    test_image_vector = test_image.reshape(1, -1)
    test_image_space = np.dot(test_image_vector - mean, eigen_vectors.T)
    distances = np.linalg.norm(face_space - test_image_space, axis=1)
    min_distance_index = np.argmin(distances)
    predicted_label = face_labels[min_distance_index]

    return predicted_label

def EigenMethod(face, mean, eigen_vectors, face_space, face_labels):

    predicted_label = EigenMethod_step2(face, mean, eigen_vectors, face_space, face_labels)

    return predicted_label

def predictLabels_ver2(img,mean, eigen_vectors, face_space, face_labels):
    from General_Functions import HaarCascade_face_detection

    marked_img, face_locations, faces = HaarCascade_face_detection(img, showResult=False)

    labels = []
    for face in faces:
        label = EigenMethod(face, mean, eigen_vectors, face_space, face_labels)
        labels.append(label)

    return labels, face_locations
#=========================================
def main():
    X,y = getData() #ofSVM
    #model,score = buildKNNmodel(X,y)
    model,score = buildSVMmodel(X,y) #ofSVM

    #mean, eigen_vectors, face_space, face_labels = EigenMethod_step1() #ofEigenFaces

    global df, data
    df = getStudentsData()

    #=============
    cap = cv2.VideoCapture(0)

    while True:
        data = []
        ret, img = cap.read()

        img = preProcess(img) #ofSVM

        labels, face_locations = predictLabels(img,model) #ofSVM
        #labels, face_locations = predictLabels_ver2(img,mean, eigen_vectors, face_space, face_labels) #ofEigenFaces

        if len(labels) > 0:
            img = showNames(img,labels,locations=face_locations)

        cv2.imshow('Look at me!', img)

        key = cv2.waitKey(10)
        if key == 13:  # --- Enter
            for row in data:
                takeAttendance(row)
        elif key == 27:  # --- ESC
            break

    cv2.destroyAllWindows()
    cap.release()
    #=============

if __name__ == '__main__':
    main()
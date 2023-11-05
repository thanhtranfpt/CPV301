'''
    Tran Xuan Thanh _ FPT University HCM.
    facebook.com/TXTPTTB
'''


def get_faces_from_folder(image_folder_path = r'K:\LEARN AI FPT\PROJECTS\images\AI1707_sampleFaces\images_rollNumber'):
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


def over_sample_face_images(image_list, label_list, output_folder = r'K:\LEARN AI FPT\PROJECTS\images\overImages\\'):
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

faces,labels = get_faces_from_folder(r"K:\LEARN AI FPT\PROJECTS\FinalCPVassignment_FaceRecognition_MachineLearning\images")
over_sample_face_images(faces,labels,r"K:\LEARN AI FPT\PROJECTS\FinalCPVassignment_FaceRecognition_MachineLearning\over_training_images\\")
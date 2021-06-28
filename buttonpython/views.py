from django.shortcuts import render

def button(request):
    return render(request,'index.html')

def firsthome(request):
    return render(request, 'first.html')

def desc(request):
    return render(request,'description.html')


def takesample(request):
    import cv2
    import requests
    import numpy as np




    face_cascade = cv2.CascadeClassifier('C:\\Users\\my pc\\PycharmProjects\\buttonpython\\cascadefiles\\haarcascade_frontalface_default.xml')


    def face_extractor(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # faces = face_cascade.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

        return img

    # url ="http://192.168.43.1:8080/shot.jpg"

    cap = cv2.VideoCapture(0);

    count = 0
    cv2.startWindowThread()
    while True:
        ret, img = cap.read()
        if face_extractor(img) is not None:
            count = count + 1
            face = cv2.resize(face_extractor(img), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            file_name_path = 'C:/Users/my pc/PycharmProjects/buttonpython/faces/user' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("face not found")
            pass

        if (cv2.waitKey(1) & 0xFF == 27 or count == 100):
            break

    cap.release()

    cv2.destroyAllWindows()

    print('collecting samples complete')

    return render(request,'index.html',{})

def train_and_recognize(request):
    import cv2
    import numpy as np
    from os import listdir
    from os.path import isfile, join

    data_path = 'C:/Users/my pc/PycharmProjects/buttonpython/faces/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    Training_Data, Labels = [], []
    ccount=1
    dataa=1

    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()

    model.train(np.asarray(Training_Data), np.asarray(Labels))

    print("Model Training Complete!!!!!")

    face_classifier = cv2.CascadeClassifier('C:\\Users\my pc\\PycharmProjects\\buttonpython\\cascadefiles\\haarcascade_frontalface_default.xml')

    def face_detector(img, size=0.5):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        if faces is ():
            return img, []

        for (x, y, w, h) in faces:
            #  nbr_predicted, conf = recognizer.predict(gray[y:y + h, x:x + w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi = img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (200, 200))

        return img, roi

    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)

        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 500:
                confidence = int(100 * (1 - (result[1]) / 300))
              #  display_string = str(confidence) + '% Confidence '
            cv2.putText(image, "matching...", (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (250, 120, 255), 2)

            if confidence > 65:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
                ccount=ccount+1

            else:
                cv2.putText(image, "No match -LOCKED", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)


        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Face Cropper', image)
            pass

        if cv2.waitKey(1) == 13 or ccount==1000:
            break

    cap.release()
    cv2.destroyAllWindows()





    return render(request,'index.html',{'dataa':dataa})


def motion_detection(request):
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    print(frame1.shape)
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 3000:
                continue
            cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)


        image = cv2.resize(frame1, (1280, 720))

        cv2.imshow("feed", frame1)
        frame1 = frame2
        ret, frame2 = cap.read()

        if cv2.waitKey(40) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


    return render(request, 'index.html', {})
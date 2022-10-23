import numpy as np
from PIL import Image
import os, cv2


# Method to train custom classifier to recognize face
def train_classifer(name):
    # Read all the images in custom data-set
    path = os.path.join(os.getcwd() + "/data/" + name + "/")

    faces = []
    ids = []
    labels = []
    pictures = {}

    # Store images in a numpy format and ids of the user on the same index in imageNp and id lists
    # 以 numpy格式存储图像，并将用户的 id存储在imageNp和 id列表中的同一索引上
    for root, dirs, files in os.walk(path):
        pictures = files

    for pic in pictures:
        imgpath = path + pic
        img = Image.open(imgpath).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(pic.split(name)[0])
        # names[name].append(id)
        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)

    # Train and save classifier 把数据标记好之后，训练模型的函数三步法
    clf = cv2.face.LBPHFaceRecognizer_create() #Ptr model = ();
    clf.train(faces, ids)   #model -> train(images, lables);
    clf.write("./data/classifiers/" + name + "_classifier.xml") #model -> save("MyFacePCAModel.xml");

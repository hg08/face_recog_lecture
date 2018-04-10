#coding:utf-8
#======================
#file:face_recog_knn.py
#Author: Huang Gang
#Date: 2018/03/30
#======================

#========
#导入模块
#========
import math
from sklearn import neighbors
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

#========
#定义函数
#========
# 计算总人数
def count(train_dir):
    """
    Counts the total number of the set.
    """
    path = train_dir
    count = 0
    for fn in os.listdir(path): #fn 表示的是文件名
            count = count + 1
    return count

# 获取所有姓名，并放于列表中
def list_all(train_dir):
    """
    Determine the list of all names.
    """
    path = train_dir
    result = []
    for fn in os.listdir(path): #fn 表示文件名
            result.append(fn)
    return result

# 训练分类模型
def train(train_dir, model_save_path='trained_knn_model.clf', n_neighbors=3, knn_algo='ball_tree', verbose=False):
    """
    训练一个用于人脸识别的k-近邻分类器.
    :参数 train_dir: 在训练文件夹中，对于每一个已知名字的人，包含一个子目录.目录名为其名字．
     (训练文件夹的结构示例：树结构)
        <训练文件夹>/
        ├── <姓名1>/
        │   ├── <照片1>.jpeg
        │   ├── <照片2>.jpeg
        │   ├── ...
        ├── <姓名2>/
        │   ├── <照片1>.jpeg
        │   └── <照片2>.jpeg
        └── ...
    :参数 model_save_path: (可选)保存的模型在硬盘上的路径
    :参数 n_neighbors: (可选)分类器中邻居的数目. 如果不制定，则自动选择
    :参数 knn_algo: (可选) 支持knn算法的底层数据结构. 默认为ball_tree
    :参数 verbose: verbosity of training
    :返回值: 根据已知数据返回一个knn分类器.
    """
    X = [] # 特征
    y = [] # 标签

    # 遍历训练集中每一个人
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # 对当前关注的人(文件夹)遍历其所有照片
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            # image 是加载成numpy数组的图片,本质是一个numpy数组
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            print('face_bounding_boxes:{}'.format(face_bounding_boxes))
            if len(face_bounding_boxes) != 1:
                # 如果当前训练照片中包含有多个人脸或者没有人脸，那么程序会跳过该照片　
                if verbose:
                    print("照片{} 不适合用于训练: {}".format(img_path, "未找到人脸" if len(face_bounding_boxes) < 1 else "发现多张人脸"))
            else:
                #　对当前照片，增加对人脸的编码to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # 决定用多少个邻居于KNN分类器中
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # 创建并训练KNN分类器
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    #保存训练所得的KNN分类器
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf


def predict(X_img_path, knn_clf=None, model_path=None, distance_threshold=0.35):
    """
    用训练好的KNN分类器来识别给定照片中的人脸
    :param X_img_path: 待识别照片的路径
    :param knn_clf: (optional) 一个knn分类器对象.　若不指定，则model_save_path 必须被指定.
    :param model_path: (optional)已经训练出的knn分类器的路径. 如果未指定, model_save_path必须是knn_clf.
    :param distance_threshold: (optional)人脸分类器的距离临界值. 其值越大,则对未知照片的错误分类的机会就越大. L: 如果训练集足够大,我们可以降低distance_thredhold的值.
    :return: 照片中＂识别出的人脸对应的名字和人脸位置＂的列表：[(name, bounding box), ...].
        对于未识别的人脸,则返回名字 'unknown'.
    """
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # 加载训练所得的KNN模型(if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # 加载照片文件并找出人脸位置
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)

    # 如果在图中未发现人脸，则返回空列表.
    if len(X_face_locations) == 0:
        return []

    # 对测试照片中的人脸编码
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

    # 对于测试照片中的人脸，用KNN模型找出最匹配的人脸
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # 做预测，并去除不在距离极限内的分类
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


# 名字列表
li_names = []

def show_prediction_labels_on_image(img_path, predictions):
    """
  　以图形来显示人脸识别结果.
    :参数img_path: 待识别照片的路径
    :参数predictions: 预测函数的结果
    :返回:
    """
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # 用Pillow模块在人脸周围画一个矩形盒子
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        # 对name编码
        name = name.encode("UTF-8")
        name = name.decode("ascii") #L add

        # 在人脸下方，显示姓名标签
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        # 添加name到列表li_names
        li_names.append(name)

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # 显示结果(照片)
    pil_image.show()

if __name__ == "__main__":
    # STEP 1:训练KNN分类器并保存至硬盘
    # 如果模型已经训练并保存,下次运行时可跳过此步(*)
    print("正在训练KNN分类器...")
    #classifier = train("knn_examples/train", model_save_path="trained_knn_model.clf", n_neighbors=2)
    print("训练完成!")

    # STEP 2: 对未知照片,运用KNN分类器做预测
    for image_file in os.listdir("knn_examples/test"):
        full_file_path = os.path.join("knn_examples/test", image_file)

        print("正在照片{}中寻找人脸".format(image_file))

        # 运用已经训练好的分类器在照片中寻找所有的人
        # 注意: You can pass in either a classifier file name or a classifier model instance
        predictions = predict(full_file_path, model_path="trained_knn_model.clf")

        # 在显示器打印结果
        for name, (top, right, bottom, left) in predictions:
            print("-在({},{})找到{}".format(left, top,name))

        # 显示结果：带有预测值的照片
        show_prediction_labels_on_image(os.path.join("knn_examples/test", image_file), predictions)

    # 对结果的分析
    s_list = set(li_names)
    s_list_all = set(list_all("knn_examples/train"))
    if "unknown" in s_list:
        s_list.remove("unknown")

    tot_num = count("knn_examples/train")
    s_absent = set(s_list_all - s_list)
    print("\n")
    print("============================\n")
    print("全体名单:",s_list_all)
    print("已到:",s_list)
    print("总人数:",tot_num)
    print("已到人数:",len(s_list))
    print("出勤率:{:.2f}".format(float(len(s_list))/float(tot_num)))
    print("未到:",s_absent)

#coding:utf-8
import sys
import dlib
from skimage import io

# 从命令行参数获取图像名称file_name
file_name = sys.argv[1]

# 建立一个HOG人脸探测器
# 方法：用内置的dlib类
face_detector = dlib.get_frontal_face_detector()

#建立窗口对象
win = dlib.image_window()

# 将图片存进一个数组
image = io.imread(file_name)

#查看image
print("image ={}".format(image))

#在图片数据上运行HOG人脸探测器.
# 结果：人脸边界盒子.
detected_faces = face_detector(image, 1)

print("发现{}张人脸于文件{}".format(len(detected_faces), file_name))

# 在桌面打开一个窗口以显示图片
win.set_image(image)

# 遍历照片中的每一张人脸
for i, face_rect in enumerate(detected_faces):

    # Detected faces are returned as an object with the coordinates
    # of the top, left, right and bottom edges
    print("- 发现人脸#{}, 位置: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

    # 在发现的每张人脸周围画一个盒子
    win.add_overlay(face_rect)

# 等待，当用户输入回车键时关闭显示图片的窗口
dlib.hit_enter_to_continue()

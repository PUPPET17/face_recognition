import face_recognition
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 加载样例图片并学习识别
def get_face_encodings(image_paths, num_jitters=10):
    encodings = []
    for path in image_paths:
        image = face_recognition.load_image_file(path)
        encodings.extend(face_recognition.face_encodings(image, num_jitters=num_jitters))
    return encodings

# 图片路径和对应姓名的列表
face_data = [
    ([r"D:\Pic\mmexport1675418256586.jpg",
      r"D:\Pic\IMG_3462.JPG"
      ], "JX"),
    ([r"D:\Pic\6b9cd86ac0eb4f33a53c02096e5d29e6~tplv-dy-aweme-images_q75.jpeg",
      r"D:\Pic\6b9cd86ac0eb4f33a53c02096e5d29e6~tplv-dy-aweme-images_q75.jpeg",
      r"D:\Pic\o8PcH3ZiVcwzPIgAjxBLT9dACiAAlB3iIEhAj~tplv-dy-aweme-images-ds-rs-v1_1440_1920_q80.JPEG",
      ], "XTC"),
    ([r"D:\Pic\7a6c68b36847406f9d4c52b588b5f76d~tplv-dy-aweme-images_q86.jpeg",
      r"D:\Pic\b7a6cf4a94cb4c1a867a383da2d817a2~tplv-dy-aweme-images_q75.jpeg",
      r"D:\Pic\82386728615241f29c94d5a93e9e83be~tplv-dy-aweme-images_q75.jpeg"
      ], "ZMF"),
    ([r"D:\Pic\IMG_0092.JPG",
      r"D:\Pic\IMG_9164.JPEG",
      r"D:\Pic\256ecd736ab346768a3f4423bf5338a9~tplv-dy-aweme-images_q75.jpeg"
      ], "HZC")
]

# 初始化已知人脸编码和姓名列表
known_encodings = []
known_names = []

# 加载每个人的多张照片并生成编码
for paths, name in face_data:
    encodings = get_face_encodings(paths)
    known_encodings.extend(encodings)
    known_names.extend([name] * len(encodings))

# 加载包含未知人脸的图片
unknown_image = face_recognition.load_image_file(r"D:\Pic\IMG_0040.JPG")

# 找到图片中的所有人脸位置和编码
face_locations = face_recognition.face_locations(unknown_image)
face_encodings = face_recognition.face_encodings(unknown_image, face_locations)

# 将图片转换为PIL格式以便使用Pillow绘制
pil_image = Image.fromarray(unknown_image)
draw = ImageDraw.Draw(pil_image)

# 循环遍历找到的每一张人脸
for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
    # 检查是否与已知人脸匹配
    matches = face_recognition.compare_faces(known_encodings, encoding)
    name = "Unknown"

    # 如果匹配上已知人脸，则取第一个匹配项
    if True in matches:
        first_match_index = matches.index(True)
        name = known_names[first_match_index]

    # 或者使用距离最小的已知人脸作为匹配
    face_distances = face_recognition.face_distance(known_encodings, encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_names[best_match_index]

    # 绘制人脸周围的矩形框
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    # 绘制人脸标签
    font = ImageFont.load_default()
    text_width, text_height = draw.textbbox((0, 0), name, font=font)[2:]  # 获取文本边界框的宽度和高度
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=font)

del draw
pil_image.show()
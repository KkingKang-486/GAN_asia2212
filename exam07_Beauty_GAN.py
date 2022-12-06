# 화장해주는 인공지능. 레퍼런스에 있는 이미지를 강도 조절하며 입힐 수 있는
# 모델 만들고 학습시키는 것 쭉 해봤는데 앱리스트 생성하는 걸 공개해뒀, 리콰이어먼츠도 있고
# 위에서부터 실행하면 실행될거라 만들어둔 GAN보델들 깃에 많음
# 모델 학습이 오래걸리는 거지 생성만 하는 건 잘됨. 만든다기보다는 갖다쓰는 것이 대부분. 깃에 올라온 모델들 갖다쓰는 것
# 뷰티 간 써볼건데 모델 필요 12:49

import dlib         # 이 친구를 깔기위해 아나콘다 > 넘파이,맥플라릿, 텐솔플로 패키지 풀로 깔고 > 환경설정도 하고 > conda install -c conda-forge dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf  # 뷰티간 모델 만든지 오래돼서 1버전으로 맞추겠다는! (빨간줄은)
tf.disable_v2_behavior() # 버전2비활성화하고
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


detector = dlib.get_frontal_face_detector()         # djfrmf
shape = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')
# 19~42 플랏 네모칸뜨게하는 거
# img = dlib.load_rgb_image('./imgs/09.jpg')
# plt.figure(figsize=(16, 10))
# plt.imshow(img)
# plt.show()                  #여기까지하면 이미지 보임 dnlcldml jwa 5ro
#
# img_result = img.copy()
# dets = detector(img, 1)
#
# # 얼굴찾아서
# if len(dets) == 0:
#     print('Not find faces')
#
# # 많이 못씀 6교시~16:00
#
# else:
#     fig, ax = plt.subplots(1, figsize=(10, 16)) # subplot (1) 1개 그려라
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height()
#         rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='b', facecolor='None')
#         # 이미지 좌표는 무조건 왼쪽 위부터 생성
#         ax.add_patch(rect)
# ax.imshow(img_result)
# plt.show()
#

# 눈, 인중, 점
# fig, ax = plt.subplots(1, figsize=(10, 6))
# obj = dlib.full_object_detections()
#
#
# for detection in dets:
#     s = shape(img, detection)
#     obj.append(s)
#
#     for point in s.parts():
#         circle = patches.Circle((point.x, point.y), radius=3, edgecolor='b', facecolor='b')
#         ax.add_patch(circle)
#     ax.imshow(img_result)
# plt.show()


# 얼굴찾아서 정렬하는 함수
def align_face(img):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = shape(img, detection)
        objs.append(s)
    faces = dlib.get_face_chips(img, objs, size=256, padding=0.35)  # 패딩 : 0.35만큼 공간을 더 줌
    return faces

#
# test_faces = align_face(img)
# fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(10, 8))
# axes[0].imshow(img)
# for i, face in enumerate(test_faces):
#     axes[i+1].imshow(face)
# plt.show()


# 7교시 : 화장입혀보겠
sess = tf.Session() #세션이란 애를 동작시켜야
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
sess.run(init_op)

saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')         # 제너레이터 = Xs

def preprocess(img):            # 이미지 전처리
    return img / 127.5 - 1
def deprocess(img):             # 제너레이터가 생성한거 다시 되돌릴 때
    return 0.5 * img + 0.5

img1 = dlib.load_rgb_image('./imgs/no_makeup/xfsy_0401.png')    #이미지 불러서 얼굴만 찾아서 정렬하는  # 다른 예시도 가능 : my_face.jpg
img1_faces = align_face(img1)

img2 = dlib.load_rgb_image('./imgs/makeup/XMY-136.png')         #gyaroo.jpg # halloween.jpg
img2_faces = align_face(img2)

# 이미지 한번 더 떠서 주석 (소스, 레퍼런스)
# fig, axes = plt.subplots(1, 2, figsize=(8, 5))
# axes[0].imshow(img1_faces[0])
# axes[1].imshow(img2_faces[0])
# plt.show()

src_img = img1_faces[0]     # 소스이미지
ref_img = img2_faces[0]     # 레퍼런스 이미지

X_img = preprocess(src_img)               #소스 이미지 전처리
X_img = np.expand_dims(X_img, axis=0)     #모델에 넣어주기 위한

Y_img = preprocess(ref_img)               #소스 이미지 전처리
Y_img = np.expand_dims(Y_img, axis=0)     #모델에 넣어주기 위한

output = sess.run(Xs, feed_dict={X:X_img, Y:Y_img})         # 뭐에 주느냐
output_img = deprocess(output[0])

# 이걸 그려보기 위해서
# 소스, 레퍼런스, 화장해서 만든 이미지 한줄에
fig, axes = plt.subplots(1, 3, figsize=(8, 5))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])
axes[2].imshow(output_img)
plt.show()



# 학습할 때 쓰기만 하는 게 아님
# 노메이크업 이미지를 원래 있던 거 말고


















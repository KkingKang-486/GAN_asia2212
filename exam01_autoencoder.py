# 별로 안중요 간(GAN)을 가기 위해 거쳐가는 / requirements 강사님이 줌에 공유해주신 걸로(전프로젝트에서?가져온. 텐솔플로 tensorflow-estimator==2.9.0)
# 깃이랑 연동 프로젝트 만든다음에 깃에 올려도 되고, 깃에서 레퍼지토리 만들고 해도되고
# 내 레퍼지토리 주소 https://github.com/KkingKang-486/GAN_asia2212.git (강사님 것 소스공유 https://github.com/scolpig/GAN_asia2212.git )
# import matplotlib.pyplot as plt
# from keras.models import *
# from keras.layers import *
# from keras.datasets import mnist # 데이터는 엠리스트에서
# # 오토인코더 # 784 → 184 → 784 이미지 압축(이미지 특성만 따로 저장 & 복원)
#
# input_img = Input(shape=(784,))             #모델부터 만들어보겠
# encoeded = Dense(32, activation='relu')     #엔코디드. 덴스레이어에 인풋 줄 것
# encoeded = encoeded(input_img)              #덴스레이어 거쳐나온 이미지(출력)
# decoded = Dense(784, activation='sigmoid')  #입력데이터를 0~1로 민맥스 정교화해서 0~1사이면 되어서 시그노이드 씀
# # 이 덴스레이어에 인코디드를 입력으로 줌 덴스레이어 두개 입력 준 것 784->32->784되도록
# decoded = decoded(encoeded)
# autoencoder = Model(input_img, decoded)
# autoencoder.summary()


import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784,))
encoded = Dense(32, activation='relu')
encoded = encoded(input_img)
decoded = Dense(784, activation='sigmoid')
decoded = decoded(encoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

encoder = Model(input_img, encoded)    # 32개
encoder.summary()   # 하나의 모델인데 밑에만 띠어서

encoder_input = Input(shape=(32,)) # 뒷부분을 떼어내야 되는데 (엔코더는 출력)디코더 입장에선 인풋
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoder_input, decoder_layer(encoder_input))
decoder.summary()   # 전에는 레이어가 많아졌는데 이제 모델 구조가 좀 더 복잡해지는.

# 압축
# 엔코더
# 디코더
# 암호화 # 규칙을 가지고 있는 게 코덱 규칙대로 움직여야하고 오토는 이 규칙을 자기가 만듬(웨이트와 바이어스로 정해질 것. 할 때마다 새로 만들 것)
# 대신 수신측에 학습된 걸 줘야하고 학습된 신호만 복원할 수 있고, 할 때마다 다시 해야함
# 별로 안중요한 모델이지만 이걸로 부터 GAN 모델이 나옴

# dkvqnqns(784->32) = 인코더, (32->784) = 디코더
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()       # 타겟은 있지만 비지도학습(자기지도학습) # 와이가 없다는 건 라벨이 필요없다는 것 언더바로 대체. 입력만 있으면 되는
x_train = x_train / 255     # 스케일링
x_test = x_test / 255     # 스케일링
# x_train /= 255              # 복합연산자  : /=

flatted_x_train = x_train.reshape(-1, 784)    # 리쉐잎해서 줄 것
flatted_x_test = x_test.reshape(-1, 784)      # 전처리까지 된 것

#이미지 제대로 복원이 되는지 보기위해
fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
                           batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

encoded_img = encoder.predict(x_test[:10].reshape(-1, 784))
decoded_img = decoder.predict(encoded_img)

n = 10
plt.gray()
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))  # 이미지 사이즈
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    #한줄에 10개씩
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])     #오류=괄호, 그래프가 떠야함
plt.show()












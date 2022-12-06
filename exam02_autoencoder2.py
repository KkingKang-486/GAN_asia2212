import matplotlib.pyplot as plt
import numpy as np
from keras.models import *
from keras.layers import *
from keras.datasets import mnist

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='sigmoid')(encoded)
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded)
autoencoder.summary()

# (784->32) = 인코더, (32->784) = 디코더
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()       # 타겟은 있지만 비지도학습(자기지도학습) # 와이가 없다는 건 라벨이 필요없다는 것 언더바로 대체. 입력만 있으면 되는
x_train = x_train / 255     # 스케일링
x_test = x_test / 255     # 스케일링

flatted_x_train = x_train.reshape(-1, 784)    # 리쉐잎해서 줄 것
flatted_x_test = x_test.reshape(-1, 784)      # 전처리까지 된 것

#이미지 제대로 복원이 되는지 보기위해
fit_hist = autoencoder.fit(flatted_x_train, flatted_x_train, epochs=50,
                           batch_size=256, validation_data=(flatted_x_test, flatted_x_test))

decoded_img = autoencoder.predict(flatted_x_test[:10])

n = 10
# plt.gray() #보라색이 디폴트
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, 10, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, 10, i + 1 + n)
    plt.imshow(decoded_img[i].reshape(28, 28))  #워드클라우드 깔면 그레이도 가능
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
    #한줄에 10개씩
plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()












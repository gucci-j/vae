# coding: utf-8
# dependencies:
# OS: macOS 10.13.3
# Python: 3.6.2
# Module: keras, tensorflow, numpy, matplotlib, h5py

from keras.layers import Dense, Flatten, Input, Reshape, Lambda, Layer
from keras.layers import Conv2D, Deconv2D, Activation
from keras import Model, metrics
from keras import backend as K
from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt

'''
# loss function layer
'''
class VAE_loss(Layer):
  def __init__(self, **kwargs):
    self.is_placeholder = True
    super(VAE_loss, self).__init__(**kwargs)

  def vae_loss(self, x, x_decoded_mean, z_sigma, z_mean):
    # クロスエントロピー
    reconst_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) 
    # 事前分布と事後分布のD_KLの値
    kl_loss = - 0.5 * K.sum(1 + K.log(K.square(z_sigma)) - K.square(z_mean) - K.square(z_sigma), axis=-1)
    return K.mean(reconst_loss + kl_loss)

  def call(self, inputs):
    x = inputs[0]
    x_decoded_mean = inputs[1]
    z_sigma = inputs[2]
    z_mean = inputs[3]
    loss = self.vae_loss(x, x_decoded_mean, z_sigma, z_mean)
    self.add_loss(loss, inputs=inputs)
    return x

class VAE(object):
    # save coefficients in advance
    # コンストラクタで定数を先に渡しておく
    def __init__(self, original_dim, latent_dim, intermediate_dim, batch_size, epsilon_std):
        self.original_dim = original_dim
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.batch_size = batch_size
        self.epsilon_std = epsilon_std

    def encoder(self):
        x = Input(shape=(self.original_dim, ))

        hidden = Dense(self.intermediate_dim, activation='relu')(x)
        z_mean = Dense(self.latent_dim, activation='linear')(hidden)
        z_sigma = Dense(self.latent_dim, activation='linear')(hidden)

        return Model(x, [z_mean, z_sigma])

    def decoder(self):
        z_mean = Input(shape=(self.latent_dim, ))
        z_sigma = Input(shape=(self.latent_dim, ))
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_sigma])
        h_decoded = Dense(self.intermediate_dim, activation='relu')(z)
        x_decoded_mean = Dense(self.original_dim, activation='sigmoid')(h_decoded)

        return Model([z_mean, z_sigma], x_decoded_mean)
    
    # サンプル生成用デコーダ
    def generator(self, _decoder):
        decoder_input = Input(shape=(self.latent_dim,))
        _, _, _, decoder_dense1, decoder_dense2 = _decoder.layers
        h_decoded = decoder_dense1(decoder_input)
        x_decoded_mean = decoder_dense2(h_decoded)

        return Model(decoder_input, x_decoded_mean)

    def sampling(self, args):
        z_mean, z_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0.,
                              stddev=self.epsilon_std)
        return z_mean + K.exp(z_sigma / 2) * epsilon

    def build_vae(self, _encoder, _decoder):
        _, encoder_dense, encoder_mean, encoder_sigma = _encoder.layers

        x = Input(shape=(self.original_dim, ))
        hidden = encoder_dense(x)
        z_mean = encoder_mean(hidden)
        z_sigma = encoder_sigma(hidden)

        self.z_m = z_mean
        self.z_s = z_sigma

        _, _, decoder_lambda, decoder_dense1, decoder_dense2 = _decoder.layers
        z = decoder_lambda([z_mean, z_sigma])
        h_decoded = decoder_dense1(z)
        x_decoded_mean = decoder_dense2(h_decoded)
        # カスタマイズした損失関数を付加する訓練用レイヤー
        y = VAE_loss()([x, x_decoded_mean, z_sigma, z_mean])

        return Model(x, y)

    def model_compile(self, model):
        model.compile(optimizer='rmsprop', loss=None)

if __name__ == '__main__':
    # coefficients 
    batch_size = 64
    original_dim = 784
    latent_dim = 2
    intermediate_dim = 256
    epochs = 1
    epsilon_std = 1.0

    '''
    # load the MNIST data
    # MNISTのデータセットを呼び出し
    '''
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = x_train.reshape((len(x_train), original_dim))
    x_test = x_test.reshape((len(x_test), original_dim))
    # 1-hot encoding
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    print('x_test shape: {0}'.format(x_test.shape))
    print('x_train shape: {0}'.format(x_train.shape))

    ''' 
    # Create an instance for the VAE model
    # VAEクラスからインスタンスを生成
    '''
    _vae = VAE(original_dim, latent_dim, intermediate_dim, batch_size, epsilon_std)
    _encoder = _vae.encoder()
    _decoder = _vae.decoder()

    '''
    # build -> compile -> summary -> fit 
    '''
    _model = _vae.build_vae(_encoder, _decoder)
    _vae.model_compile(_model)
    _model.summary()
    _hist = _model.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

    ''' 
    # weights save 
    '''
    fpath = 'vae_mnist_weights_' + str(epochs) + '.h5'
    _model.save_weights(fpath)
    
    ''' 
    # plot loss 
    '''
    loss = _hist.history['loss']
    val_loss = _hist.history['val_loss']
    plt.plot(range(1, epochs), loss[1:], marker='.', label='loss')
    plt.plot(range(1, epochs), val_loss[1:], marker='.', label='val_loss')
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    '''
    # show q(z|x) ~ p(z)
    '''
    x_test_encoded = _encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(8, 6))
    # 散布図を描画するメソッド: scatter(データx, y, 色c)
    # x_test_encoded[1]にはz_sigma:2次元データが格納されている
    plt.scatter(x_test_encoded[1][:, 0], x_test_encoded[1][:, 1], c = np.argmax(y_test, axis=1))
    plt.colorbar()
    plt.show()

    '''
    # show p(x|z) 
    '''
    n = 20  
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    grid_x = np.linspace(-2, 2, n)
    grid_y = np.linspace(-2, 2, n)
    # サンプル生成用のデコーダ: generatorを用意する
    generator = _vae.generator(_decoder)
    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            x_decoded = generator.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(figure,cmap='gray')
    plt.show()
# %%
# import
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Input, Conv2DTranspose
from tensorflow.keras.models import Model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
import tensorflow as tf
import numpy as np
import os

# %%
# load Data
file_PATH = 'images'
def load_data(file_PATH):
    file_list = []
    file_list = os.listdir(file_PATH)
    data = []
    for file in file_list:
        tmp = Image.open(os.path.join(file_PATH, file))
        data.append(np.array(tmp))
    data = np.array(data)
    data_shape = data.shape
    print(data.shape)
    return data, data_shape


data, data_shape = load_data(file_PATH)
nor_data = data / 255
data_shape[1:]

# %%
# build model
input = Input(data_shape[1:])

# encoding
x = Conv2D(32, (3, 3), padding='same', activation='relu')(input)  # 32,32,32
x = MaxPooling2D(2, 2, padding='same')(x)  # 16,16,32
x = Conv2D(8, (3, 3), padding='same', activation='relu')(x)  # 16,16,8
x = MaxPooling2D(2, 2, padding='same')(x)  # 8,8,8
x = Conv2D(1, (3, 3), padding='same', activation='relu')(x)
encoded = MaxPooling2D(2, 2, padding='same')(x)  # 8,8,8

# decodeing
x = Conv2DTranspose(1, (3, 3), padding='same', activation='relu')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(8, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2DTranspose(32, (3, 3), padding='same', activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2DTranspose(3, (3, 3), padding='same', activation='relu')(x)

# %%
# model compile
autoencoder = Model(input, decoded)
plot_model(autoencoder, show_shapes=True)
autoencoder.compile(optimizer='Adam', loss='mse')
callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1, mode='auto')
autoencoder.fit(nor_data, nor_data, batch_size=100, epochs=1000, callbacks=[callback])


# %%
# load saved model
autoencoder = keras.models.load_model('autoencoder.h5')
autoencoder.summary()
# %%
# building encoding model
encoder = Model(input, encoded)
code = encoder.predict(nor_data)
code = code.reshape((-1, 64))
# %%
#
# dimention reduction
tsne = TSNE(n_components=3, init='pca', n_iter=3000, perplexity=20)
tsne_fit = tsne.fit_transform(code)
tsne_fit[0]
# %%
# plot
tsne_df = pd.DataFrame(tsne_fit, columns=['x', 'y', 'z'])
tsne_df.head()
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="x", y="y", s='z', data=tsne_df, legend="full"
)

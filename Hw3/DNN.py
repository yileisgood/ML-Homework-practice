# %%
# import
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras import layers
print(tf.__version__)
print(tf.keras.__version__)

# %%
# load data
raw = np.genfromtxt('train.csv', delimiter=',', dtype=str, skip_header=1)
raw[0]
x_train = []
y_train = []
x_val = []
y_val = []
# tmp = np.array(raw[0, 1].split(' '), dtype=float).reshape(1, 48, 48)
# img = Image.fromarray(np.flip(tmp[0], axis=1))
# img2 = Image.fromarray(tmp[0])
# img.show()
# img2.show()
for i in range(len(raw)):
    tmp = np.array(raw[i, 1].split(' ')).reshape(1, 48, 48)
    if i % 10 == 0:
        x_val.append(tmp)
        y_val.append(raw[i, 0])
    x_train.append(tmp)
    x_train.append(np.flip(tmp, axis=2))
    y_train.append(raw[i, 0])
    y_train.append(raw[i, 0])


x_train = np.array(x_train, dtype=float) / 255.0
y_train = np.array(y_train, dtype=float) / 255.0
x_val = np.array(x_val, dtype=float)
y_val = np.array(y_val, dtype=float)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

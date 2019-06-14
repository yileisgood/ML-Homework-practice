
# %%
# import
import numpy as np
import pandas as pd
import os
# %%
# load data
x_train_raw = pd.read_csv('/users/ryan/documents/python/ml-homework-practice/hw2/x_train')
y_train_raw = pd.read_csv('/users/ryan/documents/python/ml-homework-practice/hw2/y_train')
x_test_raw = pd.read_csv('/users/ryan/documents/python/ml-homework-practice/hw2/x_test')

x_train_raw.head(5)
x_test_raw.head(5)
x_train_a = np.array(x_train_raw)
x_test_a = np.array(x_test_raw)
y_train_a = np.array(y_train_raw)
x_train_a[:, 0]
# %%
# normalization


def normalization(array):
    x_train_mean = np.mean(array, axis=0)
    x_train_std = np.std(array, axis=0)
    nor_array = np.zeros((array.shape))
    for col in range(array.shape[1]):
        nor_array[:, col] = (array[:, col] - x_train_mean[col]) / x_train_std[col]
    return nor_array


x_train_nor = normalization(x_train_a)
x_test_nor = normalization(x_test_a)
y_train_a

# %%
# compute mean and std for Guassian
class_1_id = []
class_0_id = []
for i in range(y_train_a.shape[0]):
    if y_train_a[i] == 1:
        class_1_id.append(i)
    else:
        class_0_id.append(i)
class_1 = x_train_nor[class_1_id]
class_0 = x_train_nor[class_0_id]
mean_1 = np.mean(class_1, axis=0)
mean_0 = np.mean(class_0, axis=0)
n = class_0.shape[1]
cov_0 = np.zeros((n, n))
cov_1 = np.zeros((n, n))
for i in range(class_0.shape[0]):
    cov_0 += np.dot(np.transpose([class_0[i] - mean_0]), [class_0[i] - mean_0]) / class_0.shape[0]
for i in range(class_1.shape[1]):
    cov_1 += np.dot(np.transpose([class_1[i] - mean_1]), [class_1[i] - mean_1]) / class_1.shape[1]
cov = (cov_0 * class_0.shape[0] + cov_1 * class_1.shape[0]) / (class_0.shape[0] + class_1.shape[0])
# %%
# compute Guassian using w and b
w = np.transpose((mean_0 - mean_1).dot(np.linalg.inv(cov)))
b = (-0.5) * mean_0.dot(np.linalg.inv(cov)).dot(mean_0) + (0.5) * mean_1.dot(np.linalg.inv(cov)).dot(mean_1) + np.log(class_0.shape[0] / class_1.shape[0])

# %%
# predict


def predict(data):
    prob = np.zeros((data.shape[0], 1))
    for row in range(data.shape[0]):
        z = data[row].dot(w) + b
        prob[row] = np.clip(1 / (1 + np.exp(-1 * z)), 1e-8, 1 - (1e-8))
    pred = np.zeros((data.shape[0], 1), dtype=int)
    for row in range(data.shape[0]):
        if prob[row] > 0.5:
            pred[row] = 0
    ans = np.concatenate((np.arange(1, pred.shape[0] + 1).reshape(-1, 1), pred), axis=1)
    ans = pd.DataFrame(ans, columns=['id', 'label'])
    ans.to_csv('/Users/ryan/Documents/Python/ML-Homework-practice/Hw2/answer.csv', index=False)
    return pred


pred = predict(x_test_nor)
pred.shape
np.sum(pred == y_train_a) / pred.shape[0]
# %%
# write file
ans = np.concatenate((np.arange(x_train_nor.shape[0]).reshape(-1, 1), pred), axis=1)
ans = pd.DataFrame(ans, columns=['id', 'label'])
ans.to_csv('/Users/ryan/Documents/Python/ML-Homework-practice/Hw2/answer.csv', index=False)

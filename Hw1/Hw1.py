# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
test = pd.read_csv('test.csv', header=None)
train = pd.read_csv('train.csv', encoding='big5')
output = pd.read_csv('sampleSubmission.csv')
# %%
# train
data = train.iloc[0:, 3:]
data = data.fillna(0).replace('NR', 0)
data = np.asarray(data)
switch_time = np.empty(shape=(18, 12 * 20 * 24))
for day in range(12 * 20):
    for hour in range(24):
        switch_time[:, day * 24 + hour] = data[18 * (day): 18 * (day + 1), hour]
# row for detect result, colume for each hour（at interval of 24 hour, 20 day and 12 month)

# ues every 9 hour data as one x to predict the 10th's pm value
# the shape for X should be 5652, 18*9 (include PM value of previous 9 hour)
# shape for y should be 5652, 1
# %%
num_feature = 18
X_train = np.empty(shape=(switch_time.shape[1] - 9 * 12, num_feature * 9), dtype=float)
y_train = np.empty(shape=(switch_time.shape[1] - 9 * 12, 1))

for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            X_train[month * 471 + day * 24 + hour, :] = switch_time[:, month * 480 + day * 24 + hour:month * 480 + day * 24 + hour + 9].reshape((1, -1))
            y_train[month * 471 + day * 24 + hour, 0] = switch_time[9, month * 480 + day * 24 + hour + 9]


# %%
# normalization


def normalization(X_train):
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    for i in range(X_train.shape[0]):
        for j in range(X_train.shape[1]):
            if not x_std[j] == 0:
                X_train[i][j] = (X_train[i][j] - x_mean[j]) / x_std[j]
            else:
                print(i, j)
    return X_train


# %%
# X_train = normalization(X_train)
# performance is better without normalization, might due to customized gradient

# 4 types of gradient methods
def adagrad(X_train, iteration=10000, rate=10):
    dim = X_train.shape[1] + 1
    w = np.zeros((dim, 1))
    # add bias
    X = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1).astype(float)
    learning_rate = np.array(np.ones((dim, 1)) * rate)
    adagrad_sum = np.zeros((dim, 1))
    loss_histroy = []
    for i in range(iteration):
        loss = y_train - X.dot(w)
        avg_loss = np.power(np.sum(np.power(loss, 2)) / X.shape[0], 0.5)
        grad = (-2) * X.transpose().dot(loss)
        adagrad_sum += grad ** 2
        w = w - (learning_rate * grad) / (np.sqrt(adagrad_sum) + 0.005)
        if i % (iteration / 20) == 0:
            print(avg_loss)
            loss_histroy.append(avg_loss)
    return w, loss_histroy
# %%


def l2_adagrad(X_train, iteration=50000, rate=1000, Lambda=0.000001):
    dim = X_train.shape[1] + 1
    w = np.zeros((dim, 1))
    # add bias
    X = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1).astype(float)
    learning_rate = np.array(np.ones((dim, 1)) * rate)
    adagrad_sum = np.zeros((dim, 1))
    loss_histroy = []
    for i in range(iteration):
        loss = np.power(y_train - X.dot(w), 2)
        l2 = np.power(w, 2) * Lambda
        avg_loss = np.power((np.sum(loss) + np.sum(l2)) / X.shape[0], 0.5)
        grad = (-2) * X.transpose().dot(loss)
        adagrad_sum += grad ** 2
        w = w * (1 - (Lambda * learning_rate)) - (learning_rate * grad) / (np.sqrt(adagrad_sum) + 0.005)
        if i % (iteration / 20) == 0:
            print(avg_loss)
            loss_histroy.append(avg_loss)
    return w, loss_histroy


def RMSprop(X_train, iteration=1000, rate=0.2, decay_rate=0.99):
    dim = X_train.shape[1] + 1
    w = np.zeros((dim, 1))
    r = np.zeros((dim, 1))
    # add bias
    X = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1).astype(float)
    learning_rate = np.array(np.ones((dim, 1)) * rate)
    loss_histroy = []
    for i in range(iteration):
        loss = y_train - X.dot(w)
        avg_loss = np.power(np.sum(np.power(loss, 2)) / X.shape[0], 0.5)
        grad = (-2) * X.transpose().dot(loss)
        r = r * decay_rate + ((1 - decay_rate) * np.multiply(grad, grad))
        w = w - ((learning_rate / (np.sqrt(r) + 0.0000001)) * grad)
        if i % (iteration / 20) == 0:
            print(avg_loss)
            loss_histroy.append(avg_loss)
    return w, loss_histroy


def Adam(X_train, iteration=1000, rate=0.2, decay_rate_s=0.9, decay_rate_r=0.999):
    dim = X_train.shape[1] + 1
    w = np.zeros((dim, 1))
    s = np.zeros((dim, 1))
    r = np.zeros((dim, 1))
    # add bias
    X = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1).astype(float)
    learning_rate = np.array(np.ones((dim, 1)) * rate)
    loss_histroy = []
    for i in range(iteration):
        loss = y_train - X.dot(w)
        avg_loss = np.power(np.sum(np.power(loss, 2)) / X.shape[0], 0.5)
        grad = (-2) * X.transpose().dot(loss)
        s = s * decay_rate_s + ((1 - decay_rate_s) * grad)
        r = r * decay_rate_r + ((1 - decay_rate_r) * np.multiply(grad, grad))
        delta_s = s / (1 - decay_rate_s)
        delta_r = r / (1 - decay_rate_r)
        w = w - ((learning_rate * delta_s) / (np.sqrt(delta_r) + 0.000000001))
        if i % (iteration / 20) == 0:
            print(avg_loss)
            loss_histroy.append(avg_loss)
    return w, loss_histroy

# %%


def predict(test, w, loss_histroy):
    x_test_before = test.iloc[:, 2:]
    x_test_before = x_test_before.fillna(0).replace('NR', 0)
    x_test_before = np.array(x_test_before)
    x_test = np.empty((240, num_feature * 9))
    for id in range(240):
        x_test[id, :] = x_test_before[id * num_feature:(id + 1) * num_feature, :].reshape(1, -1)
    # normalization
    xt_nor = x_test
    # xt_nor = normalization(x_test)

    test_x = np.concatenate((np.ones((xt_nor.shape[0], 1)), xt_nor), axis=1).astype(float)
    ans = test_x.dot(w)
    id_ls = []
    for id in range(240):
        id_ls.append('id_' + str(id))
    ans_ar = np.concatenate((np.asarray(id_ls).reshape(240, 1), ans), axis=1)
    ans_df = pd.DataFrame(ans_ar, columns=['id', 'value'])
    ans_df.to_csv('answer.csv', index=False)


def writecsv(pred):
    pred = pred.reshape((240, 1))
    id_ls = []
    for id in range(240):
        id_ls.append('id_' + str(id))
    ans_ar = np.concatenate((np.asarray(id_ls).reshape(240, 1), pred), axis=1)
    ans_df = pd.DataFrame(ans_ar, columns=['id', 'value'])
    ans_df.to_csv('answer.csv', index=False)


# %%
# training and predict
w_adam, loss_histroy_adam = Adam(X_train, iteration=10000, rate=0.05)
predict(test, w_adam, loss_histroy_adam)
w_rms, loss_histroy_rms = RMSprop(X_train, iteration=10000, rate=0.01, decay_rate=0.9)
predict(test, w_rms, loss_histroy_rms)

w_l2, loss_histroy_l2 = l2_adagrad(X_train, iteration=10000, Lambda=0.001)

# %%
# visualazation
plt.plot(list(np.arange(0, len(loss_histroy_adam))), loss_histroy_adam)

# %%
# sklearn practicing
# data processing
X_train
x_test_before = test.iloc[:, 2:]
x_test_before = x_test_before.fillna(0).replace('NR', 0)
x_test_before = np.array(x_test_before)
x_test = np.empty((240, num_feature * 9))
for id in range(240):
    x_test[id, :] = x_test_before[id * num_feature:(id + 1) * num_feature, :].reshape(1, -1)
# normalization
x_test
X_train_nor = normalization(X_train)
xt_nor = normalization(x_test)
test_x = np.concatenate((np.ones((xt_nor.shape[0], 1)), xt_nor), axis=1).astype(float)
y_train = y_train.reshape(y_train.shape[0],)
svr_param = [{'kernel': ['linear'], 'C':[0.4, 0.5, 0.6]}]
svr = GridSearchCV(SVR(cache_size=400), svr_param, cv=2, verbose=2, scoring='neg_mean_squared_error', n_jobs=1)
svr.fit(X_train_nor, y_train)
svr.best_score_
svr_best_param = svr.best_params_
svr_best_param['C'] = np.array(svr_best_param['C'])
svr.cv_results_['params']
svr.cv_results_['mean_test_score']
svr_best = SVR(kernel='linear', C=0.5)
svr_best.fit(X_train_nor, y_train)
svr_best.score(X_train_nor, y_train)
svr_pred = svr_best.predict(xt_nor)
writecsv(svr_pred)
lasso = linear_model.LassoCV(verbose=False, n_jobs=1, cv=None, alphas=[0, 1, 10], normalize=True)
lasso.fit(X_train_nor, y_train)
lasso.score(X_train_nor, y_train)
linear_pred = lasso.predict(xt_nor)
writecsv(linear_pred)

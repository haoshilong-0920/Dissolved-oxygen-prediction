import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from utils import *
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb

train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
input_length = ""
output_length = ""
learning_rate = 0.01
estimators = 100
max_depth = 5
interval_length = ""
scalar = True
scalar_contain_labels = True
target_value = ''

if output_length > 1:
    forecasting_model = 'multi_steps'
else:
    forecasting_model = 'one_steps'

df = pd.read_excel("", sheet_name='')
df = df[:interval_length]
features_num = ""
if features_num > 1:
    features_ = df.values
else:
    features_ = df[target_value].values
labels_ = df[target_value].values
split_train_val, split_val_test = int(len(features_)*train_ratio), \
                                  int(len(features_)*train_ratio)+int(len(features_)*val_ratio)

if scalar:
    train_val_features_ = features_[:split_val_test]
    test_features_ = features_[split_val_test:]
    scalar = preprocessing.StandardScaler()
    if features_num == 1:
        train_val_features_ = np.expand_dims(train_val_features_, axis=1)
        test_features_ = np.expand_dims(test_features_, axis=1)
    train_val_features_ = scalar.fit_transform(train_val_features_)
    test_features_ = scalar.transform(test_features_)
    features_ = np.vstack([train_val_features_, test_features_])
    if scalar_contain_labels:
        labels_ = features_[:, -1]

if len(features_.shape) == 1:
    features_ = np.expand_dims(features_, 0).T
features, labels = get_rolling_window_multistep(output_length, 0, input_length,
                                                features_.T, np.expand_dims(labels_, 0))

labels = np.squeeze(labels, axis=1)
split_train_val, split_val_test = int(len(features)*train_ratio), int(len(features)*train_ratio)+int(len(features)*val_ratio)
train_features, train_labels = features[:split_train_val], labels[:split_train_val]
val_features, val_labels = features[split_train_val:split_val_test], labels[split_train_val:split_val_test]
test_features, test_labels = features[split_val_test:], labels[split_val_test:]

train_features = train_features.reshape(len(train_features), train_features.shape[1]*train_features.shape[2])
val_features = val_features.reshape(len(val_features), val_features.shape[1]*val_features.shape[2])
test_features = test_features.reshape(len(test_features), test_features.shape[1]*test_features.shape[2])
train_features = np.concatenate((train_features, val_features), axis=0)
train_labels = np.concatenate((train_labels, val_labels), axis=0)

multioutputregressor = MultiOutputRegressor(
    lgb.LGBMRegressor(max_depth=max_depth, n_estimators=estimators,
                      learning_rate=learning_rate, force_row_wise=True, verbosity=-1))

print("——————————————————————Training Starts——————————————————————")
trained_regressor = multioutputregressor.fit(train_features, train_labels)
print("——————————————————————Training Ends——————————————————————")
print("——————————————————————Testing Starts——————————————————————")
pre_array = multioutputregressor.predict(test_features)
print("——————————————————————Testing Ends——————————————————————")
print("——————————————————————Post-Processing——————————————————————")
if scalar_contain_labels and scalar:

    pre_inverse = []
    test_inverse = []
    if features_num == 1 and output_length == 1:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice, :], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(test_labels[pre_slice, :], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)

    elif features_num > 1:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.concatenate((np.zeros((pre_array[0].shape[0], features_num-1)),np.expand_dims(pre_array[pre_slice], axis=1)), 1))[:,-1]
            test_inverse_slice = scalar.inverse_transform(np.concatenate((np.zeros((test_labels[0].shape[0], features_num-1)), np.expand_dims(test_labels[pre_slice], axis=1)), 1))[:,-1]
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse)
        test_labels = np.array(test_inverse)

    else:
        for pre_slice in range(pre_array.shape[0]):
            pre_inverse_slice = scalar.inverse_transform(np.expand_dims(pre_array[pre_slice,:], axis=1))
            test_inverse_slice = scalar.inverse_transform(np.expand_dims(test_labels[pre_slice,:], axis=1))
            pre_inverse.append(pre_inverse_slice)
            test_inverse.append(test_inverse_slice)
        pre_array = np.array(pre_inverse).squeeze(axis=-1)
        test_labels = np.array(test_inverse).squeeze(axis=-1)
    plt.figure(figsize=(40,20))
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g')
        plt.plot(test_labels[0], "r")
        plt.show()
    else:
        plt.plot(pre_array, 'g')
        plt.plot(test_labels, "r")
        plt.show()
    MSE_l = mean_squared_error(test_labels, pre_array)
    MAE_l = mean_absolute_error(test_labels, pre_array)
    MAPE_l = mean_absolute_percentage_error(test_labels, pre_array)
    R2 = r2_score(test_labels, pre_array)
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)
else:
    plt.figure(figsize=(40,20))
    if forecasting_model == 'multi_steps':
        plt.plot(pre_array[0], 'g')
        plt.plot(test_labels[0], "r")
        plt.show()
    else:
        plt.plot(pre_array, 'g')
        plt.plot(test_labels, "r")
        plt.show()
    MSE_l = mean_squared_error(test_labels, pre_array)
    MAE_l = mean_absolute_error(test_labels, pre_array)
    MAPE_l = mean_absolute_percentage_error(test_labels, pre_array)
    R2 = r2_score(test_labels, pre_array)
    print('MSE loss=%s'%MSE_l)
    print('MAE loss=%s'%MAE_l)
    print('MAPE loss=%s'%MAPE_l)
    print('R2=%s'%R2)
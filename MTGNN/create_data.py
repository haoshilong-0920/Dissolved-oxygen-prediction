import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

excel_file = '.xlsx'
sheets = pd.read_excel(excel_file, sheet_name=None)

sheet_names = list(sheets.keys())

sheet_data = [sheets[sheet_name].iloc[:, 1:].values for sheet_name in sheet_names]
original_sheet_data = [data.copy() for data in sheet_data]

scaler = StandardScaler()
for i in range(len(sheet_data)):
    sheet_data[i] = scaler.fit_transform(sheet_data[i])

window_size = ""
seq_lenth_y = ""
step_size = ""

X = []
y = []

for i in range(0, sheet_data[0].shape[0] - window_size - seq_lenth_y + 1, step_size):
    window = []

    for j in range(""):
        window.append(sheet_data[j][i:i + window_size])

    window = np.vstack(window)
    X.append(window)
    y.append(original_sheet_data[""][i + window_size:i + window_size + seq_lenth_y])

# 将数据转换为numpy数组
X = np.array(X)
y = np.array(y)
X = np.expand_dims(X, axis=-1)
y = np.expand_dims(y, axis=-1)

# 划分训练集、验证集和测试集
train_size = int(0.7 * len(X))
val_size = int(0.1 * len(X))
test_size = len(X) - train_size - val_size

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:train_size + val_size]
y_val = y[train_size:train_size + val_size]
X_test = X[train_size + val_size:]
y_test = y[train_size + val_size:]

print("X_train.shape:", X_train.shape,"y_train.shape:",y_train.shape)
print("X_val.shape:", X_val.shape,"y_val.shape:",y_val.shape)
print("X_test.shape:", X_test.shape,"y_test.shape:",y_test.shape)

# 保存为npz文件
np.savez('data/train.npz', x=X_train, y=y_train)
np.savez('data/val.npz', x=X_val, y=y_val)
np.savez('data/test.npz', x=X_test, y=y_test)


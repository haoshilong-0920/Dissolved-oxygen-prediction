import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(100)

def create_sliding_window(data, target_var, n_in=1, n_out=1, dropnan=True):
    n_vars = data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'{col}(t-{i})') for col in df.columns]
    for i in range(0, n_out):
        cols.append(df[[target_var]].shift(-i))
        if i == 0:
            names += [f'{target_var}(t)']
        else:
            names += [f'{target_var}(t+{i})']
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

excel_file = ""
sheet_names = pd.ExcelFile(excel_file).sheet_names
for sheet_name in sheet_names:
    print(f"Processing sheet: {sheet_name}")

    data = pd.read_excel(excel_file, sheet_name=sheet_name)
    data['time'] = pd.to_datetime(data['time'])
    data.set_index('time', inplace=True)

    n_in = ""
    n_out = ""
    target_var = ''
    output_dir = ""

    scaler = StandardScaler()
    target_scaler = StandardScaler()
    scaled_target = target_scaler.fit_transform(data[[target_var]])
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

    reframed = create_sliding_window(scaled_data, target_var, n_in, n_out)
    values = reframed.values

    n_train = int(0.7 * len(values))
    n_val = int(0.1 * len(values))
    train = values[:n_train, :]
    val = values[n_train:n_train + n_val, :]
    test = values[n_train + n_val:, :]

    n_features = data.shape[1]
    train_X, train_y = train[:, :n_in * n_features], train[:, -n_out:]
    val_X, val_y = val[:, :n_in * n_features], val[:, -n_out:]
    test_X, test_y = test[:, :n_in * n_features], test[:, -n_out:]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_X_tensor = torch.tensor(train_X).float().to(device).view(-1, n_in, n_features)
    train_y_tensor = torch.tensor(train_y).float().to(device)
    val_X_tensor = torch.tensor(val_X).float().to(device).view(-1, n_in, n_features)
    val_y_tensor = torch.tensor(val_y).float().to(device)
    test_X_tensor = torch.tensor(test_X).float().to(device).view(-1, n_in, n_features)
    test_y_tensor = torch.tensor(test_y).float().to(device)

    class GRU(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(GRU, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            out, _ = self.gru(x, h0)
            out = self.linear(out[:, -1, :])
            return out

    input_size = n_features
    hidden_size = 64
    num_layers = 3
    output_size = n_out

    model = GRU(input_size, hidden_size, num_layers, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 100
    batch_size = 64
    train_data = DataLoader(TensorDataset(train_X_tensor, train_y_tensor), batch_size=batch_size, shuffle=True)
    val_data = DataLoader(TensorDataset(val_X_tensor, val_y_tensor), batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        model.train()
        for seq, labels in train_data:
            optimizer.zero_grad()
            output = model(seq)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_data:
                output = model(seq)
                loss = criterion(output, labels)
                val_loss += loss.item()
        val_loss /= len(val_data)

        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {loss.item()}, Validation Loss: {val_loss}')

    model.eval()
    with torch.no_grad():
        predicted = model(test_X_tensor).cpu().numpy()
        actual = test_y_tensor.cpu().numpy()

    predicted_inverse = target_scaler.inverse_transform(predicted.reshape(-1, 1)).reshape(-1, n_out)
    actual_inverse = target_scaler.inverse_transform(actual.reshape(-1, 1)).reshape(-1, n_out)

    writer = pd.ExcelWriter(os.path.join(output_dir, f"{sheet_name}.xlsx"))
    for i in range(n_out):
        step_predictions = predicted_inverse[:, i]
        step_actuals = actual_inverse[:, i]

        rmse = math.sqrt(mean_squared_error(step_actuals, step_predictions))
        mae = mean_absolute_error(step_actuals, step_predictions)
        r2 = r2_score(step_actuals, step_predictions)
        print(f'Step {i + 1} - RMSE: {rmse}, MAE: {mae}, R2: {r2}')

        df_step = pd.DataFrame({"Prediction": step_predictions, "True Value": step_actuals})

        sheet_name = "Step" + str(i + 1)
        df_step.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()

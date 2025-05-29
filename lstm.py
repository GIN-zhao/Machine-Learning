import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

data = pd.read_csv("inGame_110852960170020340.csv")

data = data.drop(columns=["time"])

# 填充缺失值
numeric_cols = [col for col in data.columns if col.startswith(("x", "y", "blood", "gold"))]
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

behavior_cols = [f"behavior{i}" for i in range(1, 11)]
data[behavior_cols] = data[behavior_cols].fillna(3)

label_encoders = {}
for i in range(1, 11):
    le = LabelEncoder()
    data[f"behavior{i}"] = le.fit_transform(data[f"behavior{i}"])
    label_encoders[i] = le

x_cols = [f"x{i}" for i in range(1, 11)] + [f"y{i}" for i in range(1, 11)]

# 构造 5 帧的输入数据
def create_sequences(data, seq_length):
    xs, ys_coord, ys_behavior = [], [], []
    for i in range(len(data) - seq_length):
        xs.append(data.iloc[i:i+seq_length]) #.drop(columns=x_cols + behavior_cols).values)
        ys_coord.append(data.iloc[i+seq_length][x_cols].values)
        ys_behavior.append(data.iloc[i+seq_length][behavior_cols].values)
    return np.array(xs), np.array(ys_coord), np.array(ys_behavior)

seq_length = 5
X, y_coord, y_behavior = create_sequences(data, seq_length)
# 特征工程
scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

y_coord = scaler.fit_transform(y_coord.reshape(-1, y_coord.shape[-1])).reshape(y_coord.shape)

print(y_coord[:5])


X = torch.FloatTensor(X)
y_coord = torch.FloatTensor(y_coord)
y_behavior = torch.LongTensor(y_behavior)

X_train, X_test, y_coord_train, y_coord_test, y_behavior_train, y_behavior_test = train_test_split(
    X, y_coord, y_behavior, test_size=0.2, shuffle=False
)

batch_size = 32
train_dataset = TensorDataset(X_train, y_coord_train, y_behavior_train)
test_dataset = TensorDataset(X_test, y_coord_test, y_behavior_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size_coord, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc_coord = nn.Linear(hidden_size, output_size_coord)  
        self.fc_behavior = nn.Linear(hidden_size, 10 * num_classes)  

    def forward(self, x):
        # (batch_size, seq_length, input_size)
        out, (hn, cn) = self.lstm(x)
        out = out[:, -1, :] 
        coord_pred = self.fc_coord(out)  # 坐标预测
        behavior_pred = self.fc_behavior(out)  # 行为类别预测
        behavior_pred = behavior_pred.view(-1, 10, num_classes)  
        return coord_pred, behavior_pred

input_size = X_train.shape[2] 
hidden_size = 128             
output_size_coord = y_coord_train.shape[1] 
num_classes = len(label_encoders[1].classes_)  # 行为类别总数
model = LSTMModel(input_size, hidden_size, output_size_coord, num_classes)


criterion_coord = nn.MSELoss()  
criterion_behavior = nn.CrossEntropyLoss()  # 行为类别预测损失
optimizer = optim.Adam(model.parameters(), lr=0.001)


num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss_coord, train_loss_behavior = 0.0, 0.0
    for inputs, targets_coord, targets_behavior in train_loader:
        optimizer.zero_grad()
        coord_pred, behavior_pred = model(inputs)
        loss_coord = criterion_coord(coord_pred, targets_coord)

        # 计算行为类别损失
        loss_behavior = 0
        for i in range(10):  # 遍历 10 个人物的行为类别
            loss_behavior += criterion_behavior(behavior_pred[:, i], targets_behavior[:, i])
        loss_behavior /= 10 

        loss = loss_coord + loss_behavior 
        loss.backward()
        optimizer.step()
        train_loss_coord += loss_coord.item() * inputs.size(0)
        train_loss_behavior += loss_behavior.item() * inputs.size(0)
    train_loss_coord /= len(train_loader.dataset)
    train_loss_behavior /= len(train_loader.dataset)

   
    model.eval()
    test_loss_coord, test_loss_behavior = 0.0, 0.0
    with torch.no_grad():
        for inputs, targets_coord, targets_behavior in test_loader:
            coord_pred, behavior_pred = model(inputs)
            test_loss_coord += criterion_coord(coord_pred, targets_coord).item() * inputs.size(0)

            # 计算行为类别损失
            loss_behavior = 0
            for i in range(10):
                loss_behavior += criterion_behavior(behavior_pred[:, i], targets_behavior[:, i])
            loss_behavior /= 10
            test_loss_behavior += loss_behavior * inputs.size(0)
    test_loss_coord /= len(test_loader.dataset)
    test_loss_behavior /= len(test_loader.dataset)

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Coord Loss: {train_loss_coord:.4f} | Train Behavior Loss: {train_loss_behavior:.4f} | "
          f"Test Coord Loss: {test_loss_coord:.4f} | Test Behavior Loss: {test_loss_behavior:.4f}")

model.eval()
predictions_coord, true_values_coord = [], []
predictions_behavior, true_values_behavior = [], []
with torch.no_grad():
    for inputs, targets_coord, targets_behavior in test_loader:
        coord_pred, behavior_pred = model(inputs)
        predictions_coord.extend(coord_pred.numpy())
        true_values_coord.extend(targets_coord.numpy())
        predictions_behavior.extend(torch.argmax(behavior_pred, dim=2).numpy())
        true_values_behavior.extend(targets_behavior.numpy())

predictions_coord = np.array(predictions_coord)
true_values_coord = np.array(true_values_coord)
predictions_behavior = np.array(predictions_behavior)
true_values_behavior = np.array(true_values_behavior)


mse = mean_squared_error(true_values_coord, predictions_coord)
mae = mean_absolute_error(true_values_coord, predictions_coord)
print(f"Coordinate Prediction - MSE: {mse:.4f}, MAE: {mae:.4f}")

accuracy = accuracy_score(true_values_behavior.flatten(), predictions_behavior.flatten())
print(f"Behavior Prediction - Accuracy: {accuracy:.4f}")
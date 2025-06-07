import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gc

# crate train data
def generate_data(num_samples, max_num=100):
    X = []
    y = []
    for _ in range(num_samples):
        a = np.random.randint(0, max_num)
        b = np.random.randint(0, max_num)
        op = np.random.choice([0, 1])  # 0 表示加法，1 表示减法
        if op == 0:
            result = a + b
        else:
            result = a - b
        # 输入数据编码为 [a, b, op]
        X.append([a, b, op])
        y.append(result)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).view(-1, 1)

# build ai
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# start train
def train_model(model, train_X, train_y, epochs=20000, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    return model

# crate train data
train_X, train_y = generate_data(num_samples=10000)

# init model
model = SimpleNet()

# train model
trained_model = train_model(model, train_X, train_y)

# save model
model_path = 'trained_model.pth'
torch.save(trained_model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# clean
del train_X, train_y, model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# test model
test_X, test_y = generate_data(num_samples=10)
with torch.no_grad():
    predictions = trained_model(test_X)

for i in range(len(test_X)):
    a, b, op = test_X[i].numpy()
    true_result = test_y[i].item()
    pred_result = predictions[i].item()
    op_symbol = '+' if op == 0 else '-'
    print(f'{int(a)} {op_symbol} {int(b)} = {true_result:.0f}, Predicted: {pred_result:.0f}')

# clean again
del test_X, test_y, trained_model, predictions
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
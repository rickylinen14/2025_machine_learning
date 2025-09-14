import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================
# Runge function
# =============================
def runge_function(x):
    return 1 / (1 + 25 * x**2)

# =============================
# Define Neural Network (Method 1: One hidden layer, ReLU)
# =============================
class MLPReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    def forward(self, x):
        return self.model(x)

# =============================
# Training Data
# =============================
x = np.linspace(-1, 1, 200).reshape(-1, 1)
y = runge_function(x)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_val_tensor = torch.tensor(x_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# =============================
# Initialize Model
# =============================
net = MLPReLU()
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

# =============================
# Training Loop
# =============================
train_losses, val_losses = [], []
epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = net(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    val_output = net(x_val_tensor)
    val_loss = criterion(val_output, y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

# =============================
# Prediction
# =============================
x_tensor = torch.tensor(x, dtype=torch.float32)
with torch.no_grad():
    y_pred = net(x_tensor).numpy()

mse = np.mean((y - y_pred)**2)
max_error = np.max(np.abs(y - y_pred))

print("MSE:", mse)
print("Max Error:", max_error)

# =============================
# Plot Function Approximation
# =============================
plt.figure(figsize=(6, 4))
plt.plot(x, y, label="True Runge Function", linewidth=2)
plt.plot(x, y_pred, label="NN Prediction (ReLU)", linestyle="dashed")
plt.legend()
plt.title("Runge Function Approximation (ReLU)")
plt.show()

# =============================
# Plot Loss Curve
# =============================
plt.figure(figsize=(6, 4))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Training/Validation Loss")
plt.show()

# =============================
# Extra Part: Analyze ReLU and its Derivative
# =============================
relu = nn.ReLU()

# 建立輸入，要求自動微分
x_check = torch.linspace(-3, 3, 200, requires_grad=True).unsqueeze(1)
y_relu = relu(x_check)

# 我們讓 autograd 對 y_relu.sum() 求導，得到整條曲線的梯度
grad = torch.autograd.grad(outputs=y_relu, inputs=x_check,
                           grad_outputs=torch.ones_like(y_relu),
                           create_graph=False, retain_graph=False)[0]

relu_grad = grad.detach().numpy()
theoretical_grad = np.where(x_check.detach().numpy() > 0, 1.0, 0.0)

# Accuracy check (MSE between autograd vs theoretical)
grad_mse = np.mean((relu_grad - theoretical_grad)**2)
print("Accuracy of ReLU' (autograd vs theoretical):", 1 - grad_mse)

# Plot ReLU and its derivative
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(x_check.detach().numpy(), y_relu.detach().numpy())
plt.title("ReLU Function")

plt.subplot(1, 2, 2)
plt.plot(x_check.detach().numpy(), relu_grad, label="Autograd dReLU/dx")
plt.plot(x_check.detach().numpy(), theoretical_grad, linestyle="dashed", label="Theoretical dReLU/dx")
plt.legend()
plt.title("ReLU Derivative")
plt.show()

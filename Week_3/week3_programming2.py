import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================
# Runge function and derivative
# =============================
def runge_function(x):
    return 1 / (1 + 25 * x**2)

def runge_derivative(x):
    return -50 * x / (1 + 25 * x**2)**2

# =============================
# Define Neural Network
# =============================
class MLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

# =============================
# Training data
# =============================
np.random.seed(0)
torch.manual_seed(0)

X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = runge_function(X)
y_prime = runge_derivative(X)

X_train, X_val, y_train, y_val, yp_train, yp_val = train_test_split(
    X, y, y_prime, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
y_train = torch.tensor(y_train, dtype=torch.float32)
yp_train = torch.tensor(yp_train, dtype=torch.float32)

X_val = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
y_val = torch.tensor(y_val, dtype=torch.float32)
yp_val = torch.tensor(yp_val, dtype=torch.float32)

# =============================
# Initialize model, optimizer
# =============================
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =============================
# Loss history
# =============================
train_losses = []
val_losses = []

# =============================
# Training loop
# =============================
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    # Predict f(x)
    pred_f = model(X_train)

    # Compute derivative via autograd
    pred_f_grad = torch.autograd.grad(
        outputs=pred_f,
        inputs=X_train,
        grad_outputs=torch.ones_like(pred_f),
        create_graph=True
    )[0]

    # Loss: function + derivative
    loss_f = nn.MSELoss()(pred_f, y_train)
    loss_fp = nn.MSELoss()(pred_f_grad, yp_train)
    loss = loss_f + loss_fp

    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    pred_val_f = model(X_val)
    pred_val_grad = torch.autograd.grad(
        outputs=pred_val_f,
        inputs=X_val,
        grad_outputs=torch.ones_like(pred_val_f),
        create_graph=True
    )[0]

    val_loss_f = nn.MSELoss()(pred_val_f, y_val)
    val_loss_fp = nn.MSELoss()(pred_val_grad, yp_val)
    val_loss = val_loss_f + val_loss_fp

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if (epoch+1) % 200 == 0:
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {loss.item():.6f}, "
              f"Val Loss: {val_loss.item():.6f}")

# =============================
# Plot results
# =============================
X_plot = torch.linspace(-1, 1, 400).view(-1, 1)
X_plot.requires_grad_(True)

with torch.no_grad():
    y_true = runge_function(X_plot.numpy())
    y_pred = model(X_plot).numpy()

plt.figure(figsize=(6,4))
plt.plot(X_plot.detach().numpy(), y_true, label="True f(x)")
plt.plot(X_plot.detach().numpy(), y_pred, label="NN prediction f(x)", linestyle="dashed")
plt.legend()
plt.title("Runge Function Approximation")
plt.show(block=True)

plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.title("Training/Validation Loss")
plt.show(block=True)

# =============================
# Report errors
# =============================
mse_f = nn.MSELoss()(torch.tensor(y_pred), torch.tensor(y_true)).item()
max_err = np.max(np.abs(y_pred - y_true))
print(f"MSE on f(x): {mse_f:.6e}")
print(f"Max error on f(x): {max_err:.6e}")
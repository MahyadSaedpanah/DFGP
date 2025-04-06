import torch
import torch.nn as nn
import torch.optim as optim
from flatness_minima import DFGP  # 👈 اینجا کد خودت رو import کن

# مدل خیلی ساده
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

# داده ساده
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
y = torch.tensor([[1.0], [0.0]])

# loss و init
loss_fn = nn.MSELoss()
model = SimpleModel()
base_optimizer = optim.SGD(model.parameters(), lr=0.01)
doptimizer = DFGP(base_optimizer, model, rho=0.05, eta=0.01, epsilon_prime=0.01, lambda_val=0.1)

# داده پرچب‌شده (مثلاً با نویز ساده)
perturbed_x = x + 0.01 * torch.randn_like(x)
perturbed_targets = y  # چون Mixup نداریم

# ========== مرحله اصلی ==========
# ذخیره‌سازی برای چاپ دستی
original_grads = {}
hessian_approximations = {}

# مرحله اول: گرادیان اولیه
doptimizer.optimizer.zero_grad()
output = model(x)
loss = loss_fn(output, y)
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        original_grads[name] = param.grad.clone().detach()

# پرچب کردن وزن‌ها
doptimizer.perturb_step()

# مرحله دوم: گرادیان با وزن پرچب‌شده
doptimizer.optimizer.zero_grad()
output2 = model(x)
loss2 = loss_fn(output2, y)
loss2.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        g = original_grads[name]
        g_perturbed = param.grad.clone().detach()
        h_approx = (g_perturbed - g) / doptimizer.epsilon_prime
        hessian_approximations[name] = h_approx

        # چاپ نتایج
        print(f"\n🧠 لایه: {name}")
        print(f"🎯 گرادیان اولیه:\n{g}")
        print(f"💥 گرادیان بعد از پرچب:\n{g_perturbed}")
        print(f"📐 هسین تخمینی (H·v):\n{h_approx}")

# بازیابی وزن‌ها و آپدیت
doptimizer.unperturb_step()
doptimizer.step()

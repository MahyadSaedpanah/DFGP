import torch
from collections import defaultdict

class SAM:
    def __init__(self, optimizer, model, rho=0.05, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)
        self.epsilon = 1e-3

    @torch.no_grad()
    def perturb_step(self):
        # محاسبه نرم گرادیان
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        self.grad_norm = grad_norm

        # ذخیره پارامترهای و گرادیان‌های اصلی
        original_params = {}
        original_grads = {}
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                original_params[n] = p.data.clone()
                original_grads[n] = p.grad.clone()  # ∇wL'(W,X,Y)

        for n, p in self.model.named_parameters():
            if p.grad is None:
                self.state[p]["eps"] = torch.zeros_like(p)
                continue

            # محاسبه Ŵ-W
            eps = torch.clone(p.grad).detach()
            eps.mul_(self.rho / grad_norm)

            # محاسبه گرادیان در W + ε'(Ŵ-W) برای داده‌های اصلی
            p.data.add_(self.epsilon * eps)
            grad_new_orig = torch.zeros_like(p.grad)
            if p.grad is not None:
                grad_new_orig.copy_(p.grad)  # ∇wL'(W + ε'(Ŵ-W),X,Y)

            # محاسبه گرادیان در W + ε'(Ŵ-W) برای داده‌های mixup
            grad_new_mix = torch.zeros_like(p.grad)
            if p.grad is not None:
                grad_new_mix.copy_(p.grad)  # ∇wL'(W + ε'(Ŵ-W),X̃,Ỹ)

            # بازگشت به پارامترهای اصلی
            p.data.copy_(original_params[n])

            # محاسبه Hw(Ŵ-W) برای داده‌های اصلی (معادله 9)
            hessian_term_orig = (grad_new_orig - original_grads[n]) / self.epsilon

            # محاسبه Hw̃(Ŵ-W) برای داده‌های mixup (معادله 10)
            hessian_term_mix = (grad_new_mix - original_grads[n]) / self.epsilon

            # محاسبه گرادیان اصلاح شده (معادله 11)
            modified_grad = original_grads[n] + hessian_term_orig

            # اعمال معادلات 7 و 8
            eps = original_grads[n].clone()
            eps.mul_(self.rho / grad_norm)
            eps.add_(0.5 * hessian_term_orig * eps)  # برای داده‌های اصلی
            eps.add_(0.5 * hessian_term_mix * eps)   # برای داده‌های mixup

            self.state[p]["eps"] = eps
            p.add_(eps)

        self.optimizer.zero_grad()

    @torch.no_grad()
    def perturb_step_nograd(self):
        for n, p in self.model.named_parameters():
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.zeros_like(p).detach()
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def random_perturb_step(self):
        for n, p in self.model.named_parameters():
            eps = torch.randn_like(p).detach() * 0.001
            eps.mul_(self.rho)
            self.state[p]["eps"] = eps
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def unperturb_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            if "eps" in self.state[p]:
                p.sub_(self.state[p]["eps"])

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()
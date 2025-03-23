import torch
from collections import defaultdict

class SAM(torch.optim.Optimizer):
    def __init__(self, base_optimizer, model, rho=0.05):
        defaults = dict(rho=rho)
        super(SAM, self).__init__(model.parameters(), defaults)
        
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.model = model
        self.rho = rho
        self.epsilon = 1e-3

    @torch.no_grad()
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def _compute_hessian_approx(self, p, grad_orig):
        param_orig = p.data.clone()
        p.data.add_(self.epsilon * grad_orig)
        
        if hasattr(self.model, 'get_loss'):
            with torch.enable_grad():
                loss = self.model.get_loss()
                if loss is not None:
                    loss.backward(create_graph=True)
                    grad_new = p.grad.clone()
                    p.grad.data.zero_()
        else:
            grad_new = grad_orig.clone()
        
        p.data.copy_(param_orig)
        hessian_approx = (grad_new - grad_orig) / self.epsilon
        return hessian_approx

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                grad_orig = p.grad.clone()
                e_w = grad_orig * scale
                
                # محاسبه تقریب هسین و اعمال ترم مرتبه دوم
                hessian_term = self._compute_hessian_approx(p, grad_orig)
                p.grad.add_(0.5 * hessian_term * e_w)
                
                # اعمال perturbation
                p.add_(e_w)
                self.state[p]["old_p"] = p.data.clone()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]:
                    continue
                p.data = self.state[p]["old_p"]
                del self.state[p]["old_p"]

        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            with torch.enable_grad():
                closure()
        self.first_step(True)
        
        with torch.enable_grad():
            if closure is not None:
                closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                p.grad.norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm 
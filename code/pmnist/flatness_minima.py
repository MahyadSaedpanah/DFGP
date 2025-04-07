import torch
from collections import defaultdict

class SAM:
    def __init__(self, optimizer, model, rho=0.05, eta=0.01):
        self.optimizer = optimizer
        self.model = model
        self.rho = rho
        self.eta = eta
        self.state = defaultdict(dict)

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
    def perturb_step(self):
        grads = []
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
        self.grad_norm = grad_norm
        for n, p in self.model.named_parameters():
            if p.grad is None:
                self.state[p]["eps"] = torch.zeros_like(p)
                continue
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
            eps[...] = p.grad[...]
            eps.mul_(self.rho / grad_norm)
            self.state[p]["eps"] = eps
            p.add_(eps)
        self.optimizer.zero_grad()

    @torch.no_grad()
    def unperturb_step(self):
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            # print(n)
            p.sub_(self.state[p]["eps"])

    @torch.no_grad()
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()


class DFGP(SAM):
    """
    Dual Flatness-aware Gradient Projection (DFGP) optimizer
    This optimizer extends SAM with Hessian approximation via finite differences
    """
    def __init__(self, optimizer, model, rho=0.05, eta=0.01, epsilon_prime=0.01, lambda_val=0.1):
        super(DFGP, self).__init__(optimizer, model, rho, eta)
        self.epsilon_prime = epsilon_prime  # For finite difference approximation
        self.lambda_val = lambda_val  # Weight for the perturbed data loss
        
    def compute_flatness_aware_gradient(self, original_inputs, original_targets, 
                                        perturbed_inputs, perturbed_targets, 
                                        loss_fn, mixup_criterion=None):
        """
        Implements the FlatnessAwareGradient procedure (Algorithm 2) with Hessian approximation
        
        Args:
            original_inputs: Original input data
            original_targets: Original targets
            perturbed_inputs: Perturbed input data (from mixup)
            perturbed_targets: Tuple of (targets_a, targets_b) for mixup
            loss_fn: Loss function
            mixup_criterion: Mixup loss function (optional)
        """
        # Store original parameter values
        original_params = {}
        for name, param in self.model.named_parameters():
            original_params[name] = param.data.clone()
        
        # Step 1: Calculate original gradients
        self.optimizer.zero_grad()
        
        # Compute loss on original data
        output1 = self.model(original_inputs)
        orig_loss = loss_fn(output1, original_targets)
        
        # Compute loss on perturbed data
        if mixup_criterion is not None:
            targets_a, targets_b, lam = perturbed_targets
            output2 = self.model(perturbed_inputs)
            perturbed_loss = mixup_criterion(loss_fn, output2, targets_a, targets_b, lam)
            total_loss = orig_loss + self.lambda_val * perturbed_loss
        else:
            output2 = self.model(perturbed_inputs)
            perturbed_loss = loss_fn(output2, perturbed_targets)
            total_loss = orig_loss + self.lambda_val * perturbed_loss
            
        total_loss.backward()
        
        # Save original gradients
        original_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.data.clone()
        
        # Step 2: Calculate perturbed gradients for Hessian approximation
        self.optimizer.zero_grad()
        
        # Calculate perturbation vectors without applying them
        grads = []
        perturbation_vectors = {}
        for n, p in self.model.named_parameters():
            if p.grad is None:
                continue
            grads.append(torch.norm(p.grad, p=2))
        
        if grads:  # Check if grads is not empty
            grad_norm = torch.norm(torch.stack(grads), p=2) + 1.e-16
            
            for n, p in self.model.named_parameters():
                if p.grad is None:
                    continue
                eps = p.grad.clone()
                eps.mul_(self.epsilon_prime / grad_norm)  # Use epsilon_prime for finite difference
                perturbation_vectors[n] = eps
                p.data.add_(eps)  # Apply perturbation
        
        # Compute loss on original data with perturbed weights
        output1_perturbed = self.model(original_inputs)
        orig_loss_perturbed = loss_fn(output1_perturbed, original_targets)
        
        # Compute loss on perturbed data with perturbed weights
        if mixup_criterion is not None:
            targets_a, targets_b, lam = perturbed_targets
            output2_perturbed = self.model(perturbed_inputs)
            perturbed_loss_perturbed = mixup_criterion(loss_fn, output2_perturbed, targets_a, targets_b, lam)
            total_loss_perturbed = orig_loss_perturbed + self.lambda_val * perturbed_loss_perturbed
        else:
            output2_perturbed = self.model(perturbed_inputs)
            perturbed_loss_perturbed = loss_fn(output2_perturbed, perturbed_targets)
            total_loss_perturbed = orig_loss_perturbed + self.lambda_val * perturbed_loss_perturbed
            
        total_loss_perturbed.backward()
        
        # Calculate Hessian-vector products using finite differences
        for name, param in self.model.named_parameters():
            if param.grad is not None and name in original_grads:
                # Hessian-vector product approximation: (∇L(W + ϵ'δ) - ∇L(W)) / ϵ'
                if name in perturbation_vectors:
                    hessian_term = (param.grad.data - original_grads[name]) / self.epsilon_prime
                    # Update gradient with Hessian approximation: ∇L(W) + ((∇L(W + ϵ'δ) - ∇L(W)) / ϵ')
                    param.grad.data = original_grads[name] + hessian_term
                    
        # Restore original parameters
        for name, param in self.model.named_parameters():
            if name in perturbation_vectors:
                param.data.sub_(perturbation_vectors[name])
            
        # No need to call perturb_step again at the end - this was causing issues
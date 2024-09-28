import torch
import torch.nn.functional as F

class KLDivergence:
    # Choose "forward"
    def __init__(self, kl_type: str = "forward"):
        self.kl_type = kl_type

        if self.kl_type == "forward":
            self.klfn = lambda x, y: self._kl(x, y)

        elif self.kl_type == "reverse":
            self.klfn = lambda x, y: self._kl(y, x)

        elif self.kl_type == "symmetric":
            self.klfn = lambda x, y: self._kl(x, y) + self._kl(y, x)
             
    def _kl(self, x: torch.Tensor, y: torch.Tensor):
        return F.kl_div(
            input=F.log_softmax(y, dim=-1, dtype=torch.float32),
            target=F.log_softmax(x, dim=-1, dtype=torch.float32),
            log_target=True,
            reduction="mean"
        )

    def get_divergence(self, x: torch.Tensor, y: torch.Tensor):
        return self.klfn(x, y) / x.size(0)


    # logits = outputs.get("logits")
    # target_logits = self.parameter_target.forward(self.model, inputs, noise_gradient=False).get("logits")
    # logit_noise_divergence = self.logit_distance.get_divergence(target_logits, logits)
    
    # logit_noise_divergence *= self.logit_scale # Scaling parameter which may be needed for CLM
    # self.optimizer.first_step(logit_noise_divergence, zero_grad=True)


class TRAM(torch.optim.optimizer):

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(TRAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.kl_divergence = KLDivergence(kl_type="forward")


    def _grad_norm(self):
        norm = torch.norm(
                torch.stack([
                    ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2)
                    for group in self.param_groups for p in group["params"] if p.grad is not None]),
                    p=2)
        return norm


    @torch.no_grad()
    def first_step(self, logits, target_logits, zero_grad=False):
        logit_divergence = self.kl_divergence.get_divergence(logits, target_logits)
        grad_norm = self._grad_norm()
        scale = logit_divergence / (grad_norm + 1e-12)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: 
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                
                # Compute: w + e(w)
                p.add_(e_w)

        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue

                # Get back to w from w + e(w)
                p.data = self.state[p]["p_old"] 

        # Update
        self.base_optimizer.step() 
        if zero_grad: self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):

        # Closure do a full forward-backward pass
        closure = torch.enable_grad()(closure)   

        self.first_step(predicted_logits=None, target_logits=None, zero_grad=True)
        closure()
        self.second_step(zero_grad=True)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
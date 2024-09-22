import torch


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # 计算梯度范数（即所有梯度的总大小）。
        grad_norm = self._grad_norm()

        # 遍历参数组。
        for group in self.param_groups:
            # 计算缩放因子，用于调整每个参数的更新步长。
            scale = group["rho"] / (grad_norm + 1e-12)

            # 遍历参数组中的每个参数。
            for p in group["params"]:
                # 如果参数没有梯度，跳过这个参数。
                if p.grad is None: continue

                # 将当前参数值存储在state字典中，以备后续使用。
                self.state[p]["2p"] = p.data.clone()

                # 计算e_w，即将参数梯度乘以缩放因子，并考虑自适应因子。
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)

                # 将e_w加到参数上，执行“爬升到局部最大值”操作，即w + e(w)。
                p.add_(e_w)

        # 如果zero_grad为True，则将所有梯度归零。
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        # 遍历参数组。
        for group in self.param_groups:
            # 遍历参数组中的每个参数。
            for p in group["params"]:
                # 如果参数没有梯度，跳过这个参数。
                if p.grad is None: continue

                # 使用get方法避免KeyError，如果不存在键'2p'，则使用当前p.data。
                p.data = self.state[p].get("2p", p.data)

        # 执行实际的 "sharpness-aware" 更新，即基于sharpness的优化更新。
        self.base_optimizer.step()

        # 如果zero_grad为True，则将所有梯度归零。
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

import torch


class NES(object):
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def nes_grad_est(self, x, y, net, n, sigma):
        g = torch.zeros(x.size()).to(self.device)
        g = g.view(x.size()[0], -1)
        y = y.view(-1, 1)
        # y = y.to(torch.int64)
        for _ in range(n):
            u = torch.randn(x.size()).to(self.device)
            with torch.no_grad():  # 禁用梯度计算
                out1 = net(x + sigma * u)
                out2 = net(x - sigma * u)

            out1 = torch.gather(out1, 1, y)
            # pdb.set_trace()
            out2 = torch.gather(out2, 1, y)
            # print(out1.size(),u.size(),u.view(x.size()[0],-1).size())
            # print(out1[0][y],out2[0][y])
            g += out1 * u.view(x.size()[0], -1)
            g -= out2 * u.view(x.size()[0], -1)
        g = g.view(x.size())
        return -1 / (2 * sigma * n) * g

    def nes(self, x_in, y, steps, eps, TARGETED, lr, n, sigma):
        if eps == 0:
            return x_in
        x_adv = x_in.clone()
        for i in range(steps):
            # print(f'\trunning step {i+1}/{steps} ...')
            # print(net.predict(x_adv)[0][y].item())
            print(i + 1, '/', steps)
            if TARGETED:
                step_adv = x_adv - lr * torch.sign(self.nes_grad_est(x_adv, y, self.model, n, sigma))
            else:
                step_adv = x_adv + lr * torch.sign(self.nes_grad_est(x_adv, y, self.model, n, sigma))
            diff = step_adv - x_in
            diff.clamp_(-eps, eps)
            x_adv = x_in + diff
            x_adv.clamp_(0.0, 1.0)

        return x_adv

    def __call__(self, input_xi, label_or_target, targeted=False, epsilon=0.05, step=100, lr=0.01, n=10, sigma=1e-3):
        input_xi, label_or_target = input_xi.to(self.device), label_or_target.to(self.device)
        return self.nes(input_xi, label_or_target, steps=step, eps=epsilon, TARGETED=targeted, lr=lr, n=n, sigma=sigma)

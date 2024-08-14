import torch
from torchvision import transforms
from PIL import Image
import os
import time


class BoundaryAttack:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device).eval()
        self.device = device

    def orthogonal_perturbation(self, delta, prev_sample, target_sample):
        perturb = torch.randn_like(prev_sample)
        perturb = perturb / torch.norm(perturb)
        perturb *= delta * torch.mean(self.get_diff(target_sample, prev_sample))

        diff = (target_sample - prev_sample)
        diff = diff / torch.norm(diff)
        perturb -= (torch.dot(perturb.flatten(), diff.flatten()) / torch.norm(diff) ** 2) * diff

        return perturb

    def forward_perturbation(self, epsilon, prev_sample, target_sample):
        perturb = (target_sample - prev_sample)
        perturb = perturb / torch.norm(perturb)
        perturb *= epsilon
        return perturb

    def get_diff(self, sample_1, sample_2):
        return torch.norm(sample_1 - sample_2)

    def get_prediction(self, sample):
        if sample.dim() == 3:
            sample = sample.unsqueeze(0)
        with torch.no_grad():
            output = self.model(sample)
            _, predicted = torch.max(output, 1)
            return predicted.item()

    def save_image(self, tensor, label, folder):
        image = tensor.squeeze(0).cpu().clamp(0, 1)
        image = transforms.ToPILImage()(image)
        id_no = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        image.save(os.path.join("images", folder, f"{id_no}_{label}.png"))

    def boundary_attack(self, init_img, epsilon, delta, max_steps, min_diff,
                        targeted=False, target_img=None):
        initial_sample = init_img
        target_sample = target_img if targeted else None
        folder = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        os.makedirs(os.path.join("images", folder), exist_ok=True)

        attack_class = self.get_prediction(initial_sample)
        adversarial_sample = initial_sample.clone()

        n_steps = 0
        n_calls = 0

        while True:
            if not targeted:
                trial_sample = adversarial_sample + self.forward_perturbation(epsilon, adversarial_sample,
                                                                              adversarial_sample)
            else:
                trial_sample = adversarial_sample + self.forward_perturbation(epsilon, adversarial_sample,
                                                                              target_sample)

            trial_label = self.get_prediction(trial_sample)
            n_calls += 1

            if (targeted and trial_label == attack_class) or (not targeted and trial_label != attack_class):
                adversarial_sample = trial_sample
                break
            else:
                epsilon *= 0.9

        while True:
            print(f"Step #{n_steps}...")
            print("\tDelta step...")
            while True:
                trial_samples = []
                for _ in range(10):
                    if not targeted:
                        trial_sample = adversarial_sample + self.orthogonal_perturbation(delta, adversarial_sample,
                                                                                         adversarial_sample)
                    else:
                        trial_sample = adversarial_sample + self.orthogonal_perturbation(delta, adversarial_sample,
                                                                                         target_sample)

                    trial_samples.append(trial_sample)
                trial_samples = torch.cat(trial_samples)

                predictions = [self.get_prediction(ts) for ts in trial_samples]
                n_calls += 10

                if targeted:
                    success = [p == attack_class for p in predictions]
                else:
                    success = [p != attack_class for p in predictions]

                if any(success):
                    adversarial_sample = trial_samples[success.index(True)]
                    break
                else:
                    delta *= 0.9

            print("\tEpsilon step...")
            while True:
                if not targeted:
                    trial_sample = adversarial_sample + self.forward_perturbation(epsilon, adversarial_sample,
                                                                                  adversarial_sample)
                else:
                    trial_sample = adversarial_sample + self.forward_perturbation(epsilon, adversarial_sample,
                                                                                  target_sample)

                trial_label = self.get_prediction(trial_sample)
                n_calls += 1

                if (targeted and trial_label == attack_class) or (not targeted and trial_label != attack_class):
                    adversarial_sample = trial_sample
                    epsilon /= 0.5
                    break
                else:
                    epsilon *= 0.5

            n_steps += 1
            if n_steps % 10 == 0:
                self.save_image(adversarial_sample, attack_class if targeted else trial_label, folder)

            if not targeted:
                diff = self.get_diff(adversarial_sample, initial_sample).item()
            else:
                diff = self.get_diff(adversarial_sample, target_sample).item()
            print(f"Mean Squared Error: {diff}")
            print(f"Calls: {n_calls}")

            if diff <= min_diff or n_steps > max_steps:
                self.save_image(adversarial_sample, attack_class if targeted else trial_label, folder)
                break

        return adversarial_sample, diff, n_calls

    def __call__(self, init_img, epsilon=1.0, delta=0.1, max_steps=1000, min_diff=1e-3,
                 targeted=True, target_img=None):
        init_img = init_img.to(self.device)
        target_img = target_img.to(self.device) if targeted else None
        return self.boundary_attack(init_img, epsilon=epsilon, delta=delta, max_steps=max_steps,
                                    min_diff=min_diff, targeted=targeted, target_img=target_img)

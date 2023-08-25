import numpy as np
import time
import torch
import os
import sys
import math
import torch.nn as nn
import torch.nn.functional as F


class MORAAttack():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False,
                 device='cuda', decay_step='linear', float_dis=1.0, version='pgd', ensemble_pattern='softmax'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps - 1.9 * 1e-8
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.device = device
        self.scale = True
        self.decay_step = decay_step
        self.float_dis = float_dis
        self.version = version
        self.ensemble_pattern = ensemble_pattern

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)) / (
                x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    def cw_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()

        return -(x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind))

    def check_right_index(self, output, labels):
        output_index = output.argmax(dim=-1) == labels
        mask = output_index.to(dtype=torch.int8)
        mask = torch.unsqueeze(mask, -1)
        return mask

    def get_output_scale(self, output):
        std_max_out = []
        maxk = max((10,))
        # topk from big to small
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [item / 1.0 for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_sub_max_scale(self, outputs, len_output, y):
        std_max_out = [0 for i in range(len_output)]
        for i in range(len_output):
            output_i = outputs[i].clone().detach()
            output_i_sorted, ind_i_sorted = output_i.sort(dim=1)
            ind_i = (ind_i_sorted[:, -1] == y).float()
            std_max_out[i] = torch.abs(output_i_sorted[:, -1] - output_i_sorted[:, -2]).cpu().numpy()

        max_out = std_max_out[0]
        for j in range(len_output - 1):
            for k in range(len(std_max_out[j + 1])):
                if max_out[k] < std_max_out[j + 1][k]:
                    max_out[k] = std_max_out[j + 1][k]

        scale_list = [np.abs(item) / 10.0 if item > 10.0 else 1.0 for item in max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_weight(self, x, y):
        std_max_out = []
        # x.sort  from small to big
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        std_max_out.extend(
            (x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)).cpu().numpy())

        scale_list = [math.exp(self.float_dis * np.abs(item)) / math.pow(1 + math.exp(self.float_dis * np.abs(item)), 2)
                      if self.float_dis * np.abs(item) < 100 else 1e-8 for item in std_max_out]
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_sub_weight(self, x_ensemble, x_sub, y):
        std_max_out = []
        x_ensemble_sorted, ind_ensemble_sorted = x_ensemble.sort(dim=1)
        ind = (ind_ensemble_sorted[:, -1] == y).float()
        std_max_out.extend((x_sub[np.arange(x_sub.shape[0]), y] -
                            x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -1]] * (1. - ind) -
                            x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -2]] * ind).cpu().numpy())
        scale_list = [
            math.exp(self.float_dis * np.abs(item)) / math.pow(1 + math.exp(self.float_dis * np.abs(item)), 2) if
            self.float_dis * np.abs(item) < 10.0 else 1.0 / np.abs(item) for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_probablity_weight(self, x_ensemble, x_sub, y):
        std_max_out = []
        x_ensemble_sorted, ind_ensemble_sorted = x_ensemble.sort(dim=1)
        ind = (ind_ensemble_sorted[:, -1] == y).float()

        p_z_max = (x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -1]] * (1. - ind) -
                   x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -2]] * ind)
        p_y = x_sub[np.arange(x_sub.shape[0]), y]

        z_y = x_ensemble[np.arange(x_ensemble.shape[0]), y]

        z_max = (x_ensemble[np.arange(x_ensemble.shape[0]), ind_ensemble_sorted[:, -1]] * (1. - ind) -
                 x_ensemble[np.arange(x_ensemble.shape[0]), ind_ensemble_sorted[:, -2]] * ind)

        std_max_out.extend( (p_z_max*(p_y/z_y + (1-p_z_max)/z_max )).cpu().numpy() )
        scale_list = [item for item in std_max_out]

        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def attack_single_run(self, x_in, y_in):
        x_begin_batch = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y_begin_batch = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)

        if self.norm == 'Linf':
            t = 2 * torch.rand(x_begin_batch.shape).to(self.device).detach() - 1
            x_adv_begin_batch = x_begin_batch.detach() + self.eps * torch.ones(
                [x_begin_batch.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                                    t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape(
                                        [-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x_begin_batch.shape).to(self.device).detach()
            x_adv_begin_batch = x_begin_batch.detach() + self.eps * torch.ones([x_begin_batch.shape[0], 1, 1, 1]).to(
                self.device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv_begin_batch = x_adv_begin_batch.clamp(0., 1.)
        x_best_adv = x_adv_begin_batch.clone()

        if self.loss == 'ce':
            criterion_indiv = nn.CrossEntropyLoss(reduce=False, reduction='none')
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss
        elif self.loss == 'cw':
            criterion_indiv = self.cw_loss
        else:
            raise ValueError('unknowkn loss')

        x_adv_begin_batch.requires_grad_()
        grad = torch.zeros_like(x_begin_batch)
        logits = self.model(x_adv_begin_batch)[-1]
        acc_batch = logits.max(1)[1] == y_begin_batch

        robust_flags_batch = acc_batch
        num_robust_batch = torch.sum(robust_flags_batch).item()
        robust_lin_indcs_batch = robust_flags_batch.nonzero().squeeze(dim=1)
        x_adv = x_adv_begin_batch[robust_lin_indcs_batch, :]
        x = x_begin_batch[robust_lin_indcs_batch, :]
        y = y_begin_batch[robust_lin_indcs_batch]
        logits = logits[robust_lin_indcs_batch, :]

        step_size_begin = 2 * self.eps
        x_adv_old = x_adv.clone()
        adv_acc_result_batch = torch.zeros(self.n_iter, dtype=torch.float32, device=self.device)
        for i in range(self.n_iter):

            if self.decay_step == 'linear':
                step_size = step_size_begin * (1 - i / self.n_iter)
            elif self.decay_step == 'cosine':
                step_size = step_size_begin * (1 + math.cos(i / self.n_iter * math.pi)) * 0.5
            elif self.decay_step == 'cos':
                step_size = step_size_begin * math.cos(i / self.n_iter * math.pi * 0.5)
            elif self.decay_step == 'constant':
                step_size = torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0 / 255.0]).to(
                    self.device).detach().reshape([1, 1, 1, 1])

            x_adv.requires_grad_()
            grad = torch.zeros_like(x)

            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    x_adv_input = x_adv
                    outputs = self.model(x_adv_input)  # 1 forward pass (eot_iter = 1)

                    len_output = len(outputs)

                    mask_out_sub_original = [0 for i_sub in range(len_output)]
                    scale_out_sub_original = [0 for i_sub in range(len_output)]
                    weight_out_sub_original = [0 for i_sub in range(len_output)]
                    weight_out_sub_probility = [0 for i_sub in range(len_output)]
                    out_adv_scale, out_adv_old = outputs[-2], outputs[-1]
                    mask_out_adv = self.check_right_index(out_adv_old, y)
                    mask_out_adv_grad = torch.unsqueeze(torch.unsqueeze(mask_out_adv.clone(), -1), -1)  # #############
                    scale_output_old = self.get_output_scale(out_adv_scale.clone().detach())
                    sub_output = len_output - 1

                    sub_max_out_scale = self.get_sub_max_scale(outputs, sub_output, y)

                    scale_sub_output = [0 for i_sub in range(sub_output)]
                    outputs_after_scale = 0
                    for i_sub in range(sub_output):
                        scale_sub_output[i_sub] = outputs[i_sub] / sub_max_out_scale

                    for i_sub in range(sub_output - 1):
                        outputs_after_scale += F.softmax(scale_sub_output[i_sub].clone().detach(), dim=-1)
                    outputs_after_scale = outputs_after_scale / sub_output

                    for i_pre in range(sub_output):
                        scale_out_sub_original[i_pre] = self.get_output_scale(outputs[i_pre].clone().detach())
                        mask_out_sub_original[i_pre] = self.check_right_index(scale_sub_output[i_pre], y)
                        weight_out_sub_original[i_pre] = self.get_sub_weight(out_adv_scale.clone().detach(),
                                                                             scale_sub_output[i_pre].clone().detach(),
                                                                             y)

                    sub_after_softmax = [0 for i_sub in range(sub_output)]
                    for i_pre in range(sub_output):
                        sub_after_softmax[i_pre] = F.softmax(self.float_dis * outputs[i_pre], dim=-1)
                        weight_out_sub_probility[i_pre] = self.get_probablity_weight(
                            outputs_after_scale.clone().detach(), sub_after_softmax[i_pre].clone().detach(), y)

                    scale = self.scale_value

                    if self.ensemble_pattern == 'softmax' or self.ensemble_pattern=='voting':
                        logits_prev = (1 - scale) * out_adv_scale / scale_output_old
                        loss_sub = 0
                        weight_sub = 1e-8
                        for i_pre in range(sub_output):
                            loss_sub += scale_sub_output[i_pre] * weight_out_sub_probility[i_pre] * \
                                        mask_out_sub_original[i_pre]
                            weight_sub += weight_out_sub_probility[i_pre] * mask_out_sub_original[i_pre]
                        logits_prev += scale * (loss_sub / weight_sub)

                    elif self.ensemble_pattern == 'logits':
                        logits_prev = (1 - scale) * out_adv_scale / scale_output_old
                        loss_sub = 0
                        weight_sub = 1e-8
                        for i_pre in range(sub_output):
                            loss_sub += scale_sub_output[i_pre] * weight_out_sub_probility[i_pre]
                            weight_sub += weight_out_sub_probility[i_pre]
                        logits_prev += scale * (loss_sub / weight_sub)

                    loss_indiv_prev = criterion_indiv(logits_prev, y)
                    loss_prev = loss_indiv_prev.sum()
                    logits = out_adv_old
                grad += torch.autograd.grad(loss_prev, [x_adv])[0].detach()

            grad /= float(self.eot_iter)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    if self.version == 'standard-t-pre' or self.version == 'adaptive_attack':
                        x_adv_1 = x_adv + mask_out_adv_grad * step_size * torch.sign(grad)
                        x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)

                        x_adv_1 = torch.clamp(torch.min(
                            torch.max(x_adv + mask_out_adv_grad * ((x_adv_1 - x_adv) * a + grad2 * (1 - a)),
                                      x - self.eps),
                            x + self.eps), 0.0, 1.0)
                    else:
                        x_adv_1 = x_adv + step_size * torch.sign(grad)
                        x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)

                        x_adv_1 = torch.clamp(torch.min(
                            torch.max(x_adv + ((x_adv_1 - x_adv) * a + grad2 * (1 - a)),
                                      x - self.eps),
                            x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.
            logits_after_attack = self.model(x_adv)[-1]
            pred = logits_after_attack.detach().max(1)[1] == y
            false_batch = ~ pred
            right_batch = pred
            non_robust_lin_idcs = robust_lin_indcs_batch[false_batch]
            robust_flags_batch[non_robust_lin_idcs] = False
            acc_batch[non_robust_lin_idcs] = False
            x_best_adv[non_robust_lin_idcs] = x_adv[false_batch]
            robust_lin_indcs_batch = robust_flags_batch.nonzero().squeeze(dim=1)
            x_adv_middle = x_adv[right_batch, :]
            x_adv = x_adv_middle
            x_adv_old = x_adv_old[right_batch, :]
            grad = grad[right_batch, :]
            x = x_begin_batch[robust_lin_indcs_batch, :]
            y = y_begin_batch[robust_lin_indcs_batch]
            num_robust_batch = torch.sum(robust_flags_batch).item()
            if num_robust_batch == 0:
                break
            adv_acc_result_batch[i] += num_robust_batch

        return acc_batch, x_best_adv, adv_acc_result_batch

    def perturb(self, x_in, y_in, best_loss=False, cheap=True, scale=0):
        self.scale_value = scale
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone().unsqueeze(0) if len(x_in.shape) == 3 else x_in.clone()
        y = y_in.clone().unsqueeze(0) if len(y_in.shape) == 0 else y_in.clone()

        adv = x.clone()
        x_input = x
        acc = self.model(x_input)[-1].max(1)[1] == y
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)
        adv_acc_result_batch = torch.zeros(self.n_iter * self.n_restarts, dtype=torch.float32, device=self.device)
        ind_to_fool = acc.nonzero().squeeze()
        if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
        if ind_to_fool.numel() != 0:
            x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
            acc_curr, adv_curr, adv_acc_result_batch_every_counter = self.attack_single_run(x_to_fool, y_to_fool)
            ind_curr = (acc_curr == 0).nonzero().squeeze()
            for i in range(len(adv_acc_result_batch_every_counter)):
                adv_acc_result_batch[0 * self.n_iter + i] = adv_acc_result_batch_every_counter[i]
            acc[ind_to_fool[ind_curr]] = 0
            adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
            if self.verbose:
                print('restart {} - robust accuracy: {:.2%} - cum. time: {:.1f} s'.format(
                    counter, acc.float().mean(), time.time() - startt))

        return acc, adv, adv_acc_result_batch

class MORAAttack_targeted():
    def __init__(self, model, n_iter=100, norm='Linf', n_restarts=1, eps=None,
                 seed=0, loss='ce', eot_iter=1, rho=.75, verbose=False, device='cuda',
                 n_target_classes=9, decay_step='linear', float_dis=1.0, version='pgd', ensemble_pattern='softmax'):
        self.model = model
        self.n_iter = n_iter
        self.eps = eps - 1.9 * 1e-8
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.verbose = verbose
        self.target_class = None
        self.device = device
        self.n_target_classes = n_target_classes
        self.loss = loss
        self.scale = True
        self.decay_step = decay_step
        self.float_dis = float_dis
        self.version = version
        self.ensemble_pattern = ensemble_pattern

    def check_right_index(self, output, labels):
        output_index = output.argmax(dim=-1) == labels
        mask = output_index.to(dtype=torch.int8)
        mask = torch.unsqueeze(mask, -1)
        return mask

    def dlr_loss_targeted(self, x, y, y_target):
        x_sorted, ind_sorted = x.sort(dim=1)

        return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (
                x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)

    def ce_targeted(self, x, y, y_target):
        criterion = nn.CrossEntropyLoss(reduce=False, reduction='none')
        return -criterion(x, y_target)

    def get_output_scale(self, output):
        std_max_out = []
        maxk = max((10,))
        # topk from big to small
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [np.abs(item) / 1.0 for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_sub_max_scale(self, outputs, len_output, y):
        std_max_out = [0 for i in range(len_output)]
        for i in range(len_output):
            output_i = outputs[i].clone().detach()
            output_i_sorted, ind_i_sorted = output_i.sort(dim=1)
            ind_i = (ind_i_sorted[:, -1] == y).float()
            std_max_out[i] = torch.abs(output_i_sorted[:, -1] - output_i_sorted[:, -2]).cpu().numpy()
        max_out = std_max_out[0]
        for j in range(len_output - 1):
            for k in range(len(std_max_out[j + 1])):
                if max_out[k] < std_max_out[j + 1][k]:
                    max_out[k] = std_max_out[j + 1][k]

        scale_list = [np.abs(item) / 10.0 if item > 10.0 else 1.0 for item in max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_weight(self, x, y):
        std_max_out = []
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        std_max_out.extend(
            (x[np.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1. - ind)).cpu().numpy())
        scale_list = [
            math.exp(self.float_dis * np.abs(item)) / math.pow(1 + math.exp(self.float_dis * np.abs(item)), 2) if
            self.float_dis * np.abs(item) < 10.0 else 1.0 / np.abs(item) for item in std_max_out]
        # scale_list = [1.0 for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_sub_weight(self, x_ensemble, x_sub, y):
        std_max_out = []
        x_ensemble_sorted, ind_ensemble_sorted = x_ensemble.sort(dim=1)
        ind = (ind_ensemble_sorted[:, -1] == y).float()
        std_max_out.extend((x_sub[np.arange(x_sub.shape[0]), y] -
                            x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -1]] * (1. - ind) -
                            x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -2]] * ind).cpu().numpy())
        scale_list = [1.0 / math.exp(self.float_dis * item) if self.float_dis * np.abs(item) < 100 else 1e-40 for item
                      in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def get_probablity_weight(self, x_ensemble, x_sub, y, y_target):
        std_max_out = []
        x_ensemble_sorted, ind_ensemble_sorted = x_ensemble.sort(dim=1)
        ind = (ind_ensemble_sorted[:, -1] == y).float()

        p_z_max = (x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -1]] * (1. - ind) -
                   x_sub[np.arange(x_sub.shape[0]), ind_ensemble_sorted[:, -2]] * ind)
        p_y = x_sub[np.arange(x_sub.shape[0]), y]

        std_max_out.extend((p_z_max * (p_y + 1 - p_z_max)).cpu().numpy())

        scale_list = [item for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device)
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def attack_single_run(self, x_in, y_in):
        x = x_in.clone().unsqueeze(0) if len(x_in.shape) == 3 else x_in.clone()
        y = y_in.clone().unsqueeze(0) if len(y_in.shape) == 0 else y_in.clone()

        if self.norm == 'Linf':
            t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        elif self.norm == 'L2':
            t = torch.randn(x.shape).to(self.device).detach()
            x_adv = x.detach() + self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * t / (
                    (t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        x_adv = torch.clamp(torch.min(torch.max(x_adv, x - self.eps), x + self.eps), 0.0, 1.0)
        x_adv = x_adv.clamp(0., 1.)
        x_best_adv = x_adv.clone()

        x_input = x
        output = self.model(x_input)
        y_target = output[-1].sort(dim=1)[1][:, -self.target_class]

        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        if self.loss == 'ce':
            criterion_indiv = self.ce_targeted
        elif self.loss == 'dlr':
            criterion_indiv = self.dlr_loss_targeted
        else:
            raise ValueError('unknowkn loss')

        acc = self.model(x)[-1].max(1)[1] == y

        step_size_begin = self.eps * torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor(
            [2.0]).to(self.device).detach().reshape([1, 1, 1, 1])

        x_adv_old = x_adv.clone()

        for i in range(self.n_iter):
            if self.decay_step == 'linear':
                step_size = step_size_begin * (1 - i / self.n_iter)
            elif self.decay_step == 'cosine':
                step_size = step_size_begin * (1 + math.cos(i / self.n_iter * math.pi)) * 0.5
            elif self.decay_step == 'cos':
                step_size = step_size_begin * math.cos(i / self.n_iter * math.pi * 0.5)
            elif self.decay_step == 'constant':
                step_size = torch.ones([x.shape[0], 1, 1, 1]).to(self.device).detach() * torch.Tensor([2.0 / 255.0]).to(
                    self.device).detach().reshape([1, 1, 1, 1])
            ### get gradient
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                with torch.enable_grad():
                    x_adv_input = x_adv
                    outputs = self.model(x_adv_input)  # 1 forward pass (eot_iter = 1)

                    len_output = np.int(len(outputs))
                    sub_output = len_output - 1

                    mask_out_sub_original = [0 for i in range(len_output)]
                    scale_out_sub_original = [0 for i in range(len_output)]
                    weight_out_sub_original = [0 for i in range(len_output)]
                    weight_out_sub_probility = [0 for i in range(len_output)]
                    out_adv_scale, out_adv_old = outputs[-2], outputs[-1]
                    mask_out_adv = self.check_right_index(out_adv_old, y)
                    mask_out_adv_grad = torch.unsqueeze(torch.unsqueeze(mask_out_adv.clone(), -1), -1)  # #############
                    scale_output_old = self.get_output_scale(out_adv_scale.clone().detach())

                    sub_max_out_scale = self.get_sub_max_scale(outputs, sub_output, y)

                    scale_sub_output = [0 for i in range(sub_output)]
                    outputs_after_scale = 0
                    for i in range(sub_output):
                        scale_sub_output[i] = outputs[i] / sub_max_out_scale

                    for i in range(sub_output - 1):
                        outputs_after_scale += F.softmax(scale_sub_output[i].clone().detach(), dim=-1)
                    outputs_after_scale = outputs_after_scale / sub_output

                    for i_pre in range(sub_output):
                        scale_out_sub_original[i_pre] = self.get_output_scale(outputs[i_pre].clone().detach())

                        mask_out_sub_original[i_pre] = self.check_right_index(scale_sub_output[i_pre], y)

                        weight_out_sub_original[i_pre] = self.get_sub_weight(out_adv_scale.clone().detach(),
                                                                             scale_sub_output[i_pre].clone().detach(),
                                                                             y)

                    # ##### get probability weight
                    sub_after_softmax = [0 for i in range(sub_output)]
                    for i_pre in range(sub_output):
                        sub_after_softmax[i_pre] = F.softmax(self.float_dis * outputs[i_pre], dim=-1)
                        weight_out_sub_probility[i_pre] = self.get_probablity_weight(
                            outputs_after_scale.clone().detach(), sub_after_softmax[i_pre].clone().detach(), y,
                            y_target)

                    scale = self.scale_value

                    if self.ensemble_pattern == 'softmax' or self.ensemble_pattern == 'voting':
                        logits_prev = (1 - scale) * out_adv_scale / scale_output_old
                        loss_sub = 0
                        weight_sub = 1e-8
                        for i_pre in range(sub_output):
                            loss_sub += scale_sub_output[i_pre] * weight_out_sub_probility[i_pre] * \
                                        mask_out_sub_original[i_pre]
                            weight_sub += weight_out_sub_probility[i_pre] * mask_out_sub_original[i_pre]
                        logits_prev += scale * (loss_sub / weight_sub)

                    elif self.ensemble_pattern == 'logits':
                        logits_prev = (1 - scale) * out_adv_scale / scale_output_old
                        loss_sub = 0
                        weight_sub = 1e-8
                        for i_pre in range(sub_output):
                            loss_sub += scale_sub_output[i_pre] * weight_out_sub_probility[i_pre]
                            weight_sub += weight_out_sub_probility[i_pre]
                        logits_prev += scale * (loss_sub / weight_sub)

                    logits_prev = out_adv_old
                    loss_indiv_prev = criterion_indiv(logits_prev, y, y_target)
                    loss_prev = loss_indiv_prev.sum()
                    logits = out_adv_old

                grad += torch.autograd.grad(loss_prev, [x_adv])[0].detach()  # 1 backward pass (eot_iter = 1)

            grad /= float(self.eot_iter)
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    x_adv_1 = x_adv + mask_out_adv_grad * step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - self.eps), x + self.eps), 0.0, 1.0)

                    x_adv_1 = torch.clamp(torch.min(
                        torch.max(x_adv + mask_out_adv_grad * ((x_adv_1 - x_adv) * a + grad2 * (1 - a)),
                                  x - self.eps),
                        x + self.eps), 0.0, 1.0)

                elif self.norm == 'L2':
                    x_adv_1 = x_adv + step_size[0] * grad / (
                            (grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (
                            ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        self.eps * torch.ones(x.shape).to(self.device).detach(),
                        ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                x_adv = x_adv_1 + 0.
            logits_after_attack = self.model(x_adv)[-1]
            pred = logits_after_attack.detach().max(1)[1] == y
            acc = torch.min(acc, pred)

            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        x_best_adv[(pred == 1).nonzero().squeeze()] = x_adv[(pred == 1).nonzero().squeeze()] + 0.

        return acc, x_best_adv

    def perturb(self, x_in, y_in, best_loss=False, cheap=True, scale=0):
        self.scale_value = scale
        assert self.norm in ['Linf', 'L2']
        x = x_in.clone().unsqueeze(0) if len(x_in.shape) == 3 else x_in.clone()
        y = y_in.clone().unsqueeze(0) if len(y_in.shape) == 0 else y_in.clone()

        adv = x.clone()
        x_input = x
        acc = self.model(x_input)[-1].max(1)[1] == y
        if self.verbose:
            print('-------------------------- running {}-attack with epsilon {:.4f} --------------------------'.format(
                self.norm, self.eps))
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        startt = time.time()

        torch.random.manual_seed(self.seed)
        torch.cuda.random.manual_seed(self.seed)

        for target_class in range(2, self.n_target_classes + 2):
            self.target_class = target_class
            ind_to_fool = acc.nonzero().squeeze()
            if len(ind_to_fool.shape) == 0: ind_to_fool = ind_to_fool.unsqueeze(0)
            if ind_to_fool.numel() != 0:
                x_to_fool, y_to_fool = x[ind_to_fool].clone(), y[ind_to_fool].clone()
                acc_curr, adv_curr = self.attack_single_run(x_to_fool, y_to_fool)
                ind_curr = (acc_curr == 0).nonzero().squeeze()
                #
                acc[ind_to_fool[ind_curr]] = 0
                adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                if self.verbose:
                    print(
                        'restart {} - target_class {} - robust accuracy: {:.2%} at eps = {:.5f} - cum. time: {:.1f} s'.format(
                            counter, self.target_class, acc.float().mean(), self.eps, time.time() - startt))

        return acc, adv

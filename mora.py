import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import argparse
import time
import sys
# sys.path.insert(0, '..')
from mora_attack import MORAAttack
from other_utils import Logger


class MORA():
    def __init__(self, model, norm='Linf', eps=.3, seed=None, verbose=True,
                 attacks_to_run=[], version='standard',
                 device='cuda', log_path=None, scale=True, decay_step='linear',
                 scale_value=0.9, n_iter=100, float_dis=1.0,
                 model_name='dverge', ensemble_pattern='softmax'):
        self.model = model
        self.norm = norm
        assert norm in ['Linf', 'L2']
        self.epsilon = eps
        self.seed = seed
        self.verbose = verbose
        self.attacks_to_run = attacks_to_run
        self.version = version
        self.device = device
        self.logger = Logger(log_path)
        self.scale = scale
        self.decay_step = decay_step
        self.scale_value = scale_value
        self.n_iter = n_iter
        self.float_dis = float_dis
        self.model_name = model_name
        self.ensemble_pattern = ensemble_pattern

        self.mora = MORAAttack(self.model, n_restarts=5, n_iter=self.n_iter, verbose=False,
                               eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75, seed=self.seed,
                               device=self.device, decay_step=self.decay_step, float_dis=self.float_dis,
                               version=self.version, ensemble_pattern=self.ensemble_pattern)

        from mora_attack import MORAAttack_targeted
        self.mora_targeted = MORAAttack_targeted(self.model, n_restarts=1, n_iter=self.n_iter, verbose=False,
                                                 eps=self.epsilon, norm=self.norm, eot_iter=1, rho=.75,
                                                 seed=self.seed, device=self.device, decay_step=self.decay_step,
                                                 float_dis=self.float_dis, version=self.version,
                                                 ensemble_pattern=self.ensemble_pattern)

        if version in ['standard-pre', 'standard-t-pre']:
            self.set_version(version)

    def get_logits(self, x):
        x_input = x
        return self.model(x_input)

    def get_seed(self):
        return time.time() if self.seed is None else self.seed

    def save_numpy_data_to_txt(self, name, data_array):
        filename = './loss_converge/results_' + str(self.version) + '_' + str(name) + '_'+str(self.ensemble_pattern)+'.txt'
        f = open(filename, 'a')
        np.savetxt(f, data_array, fmt='%f', delimiter=',')
        f.close()

    def run_standard_evaluation(self, x_orig, y_orig, bs=250):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                                                         ', '.join(self.attacks_to_run)))

        with torch.no_grad():
            # calculate accuracy
            n_batches = int(np.ceil(x_orig.shape[0] / bs))
            robust_flags = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
            first_minus_second_value = torch.zeros(x_orig.shape[0], dtype=torch.float, device=x_orig.device)
            for batch_idx in range(n_batches):
                start_idx = batch_idx * bs
                end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])

                x = x_orig[start_idx:end_idx, :].clone().to(self.device)
                y = y_orig[start_idx:end_idx].clone().to(self.device)
                output = self.get_logits(x)
                correct_batch = y.eq(output[-1].max(dim=1)[1])
                robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
                maxk = max((10,))
                pred_val_out, pred_id_out = output[-1].topk(maxk, 1, True, True)
                first_minus_second_value_batch = (pred_val_out[:, 0] - pred_val_out[:, 1]).detach().to(
                    first_minus_second_value.device)
                first_minus_second_value[start_idx:end_idx] = first_minus_second_value_batch

            robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]

            if self.verbose:
                self.logger.log('initial accuracy: {:.2%}'.format(robust_accuracy))
            robust_accuracy_array=[robust_accuracy]
            # self.save_numpy_data_to_txt(self.model_name, robust_accuracy_array)
            x_adv = x_orig.clone().detach()
            startt = time.time()
            num_attack = 0
            for attack in self.attacks_to_run:
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, x_orig.shape[0])

                    x = x_adv[start_idx:end_idx, :].clone().to(self.device)
                    y = y_orig[start_idx:end_idx].clone().to(self.device)
                    output = self.get_logits(x)
                    correct_batch = y.eq(output[-1].max(dim=1)[1])
                    robust_flags[start_idx:end_idx] = correct_batch.detach().to(robust_flags.device)
                    maxk = max((10,))
                    pred_val_out, pred_id_out = output[-1].topk(maxk, 1, True, True)
                    first_minus_second_value_batch = (pred_val_out[:, 0] - pred_val_out[:, 1]).detach().to(
                        first_minus_second_value.device)
                    first_minus_second_value[start_idx:end_idx] = first_minus_second_value_batch
                num_robust = torch.sum(robust_flags).item()

                if num_robust == 0:
                    break
                num_attack += 1
                if num_attack >= 3:
                    num_robust_reduce = np.min([num_robust, 10000])
                    n_batches = int(np.ceil(num_robust_reduce / bs))
                else:
                    n_batches = int(np.ceil(num_robust / bs))

                before_sorted_robust_lin_idcs = torch.nonzero(robust_flags, as_tuple=False)

                first_minus_second_value_robust = first_minus_second_value[before_sorted_robust_lin_idcs]
                sorted_first_minus_second_value, indices_first_minus_second_value = torch.sort(
                    first_minus_second_value_robust, dim=0)
                sorted_robust_lin_idcs = before_sorted_robust_lin_idcs[indices_first_minus_second_value]
                robust_lin_idcs = sorted_robust_lin_idcs

                if num_robust > 1:
                    robust_lin_idcs.squeeze_()

                for batch_idx in range(n_batches):
                    start_idx = batch_idx * bs
                    end_idx = min((batch_idx + 1) * bs, num_robust)

                    batch_datapoint_idcs = robust_lin_idcs[start_idx:end_idx]
                    if len(batch_datapoint_idcs.shape) > 1:
                        batch_datapoint_idcs.squeeze_(-1)
                    x = x_orig[batch_datapoint_idcs, :].clone().to(self.device)
                    y = y_orig[batch_datapoint_idcs].clone().to(self.device)

                    if len(x.shape) == 3:
                        x = x.unsqueeze(dim=0)
                    elif len(x.shape) == 5:
                        x = x.squeeze(dim=0)
                        y = y.squeeze(dim=0)

                    if attack[:7] == 'mora-ce':
                        scale_17_value = float(attack[8:]) / 10
                        self.mora.loss = 'ce'
                        self.mora.seed = self.get_seed()
                        _, adv_curr, adv_acc_result_batch = self.mora.perturb(x, y, cheap=True, scale=scale_17_value)

                    elif attack[:9] == 'mora_ce_t':
                        # targeted mora
                        scale_17_value = float(attack[10:]) / 10
                        self.mora.loss = 'ce'
                        self.mora_targeted.seed = self.get_seed()
                        _, adv_curr = self.mora_targeted.perturb(x, y, cheap=True, scale=scale_17_value)


                    else:
                        raise ValueError('Attack not supported')

                    output = self.get_logits(adv_curr)
                    false_batch = ~y.eq(output[-1].max(dim=1)[1]).to(robust_flags.device)
                    non_robust_lin_idcs = batch_datapoint_idcs[false_batch]
                    robust_flags[non_robust_lin_idcs] = False

                    x_adv[non_robust_lin_idcs] = adv_curr[false_batch].detach().to(x_adv.device)

                    if self.verbose:
                        num_non_robust_batch = torch.sum(false_batch)
                        self.logger.log('{} - {}/{} - {} out of {} successfully perturbed'.format(
                            attack, batch_idx + 1, n_batches, num_non_robust_batch, x.shape[0]))
                    if batch_idx == 0:
                        adv_acc_result = torch.zeros(len(adv_acc_result_batch), dtype=torch.float32,
                                                     device=self.device)
                    for i in range(len(adv_acc_result_batch)):
                        adv_acc_result[i] += adv_acc_result_batch[i]

                # self.save_numpy_data_to_txt(self.model_name, adv_acc_result.cpu().numpy()/ x_orig.shape[0])
                robust_accuracy = torch.sum(robust_flags).item() / x_orig.shape[0]
                if self.verbose:
                    self.logger.log('robust accuracy after {}: {:.2%} (total time {:.1f} s)'.format(
                        attack.upper(), robust_accuracy, time.time() - startt))

            # final check
            if self.verbose:
                if self.norm == 'Linf':
                    res = (x_adv - x_orig).abs().view(x_orig.shape[0], -1).max(1)[0]
                elif self.norm == 'L2':
                    res = ((x_adv - x_orig) ** 2).view(x_orig.shape[0], -1).sum(-1).sqrt()
                self.logger.log('max {} perturbation: {:.5f}, nan in tensor: {}, max: {:.5f}, min: {:.5f}'.format(
                    self.norm, res.max(), (x_adv != x_adv).sum(), x_adv.max(), x_adv.min()))
                self.logger.log('robust accuracy: {:.2%}'.format(robust_accuracy))

        return x_adv

    def clean_accuracy(self, x_orig, y_orig, bs=250):
        n_batches = x_orig.shape[0] // bs
        acc = 0.
        for counter in range(n_batches):
            x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(self.device)
            output = self.get_logits(x)
            acc += (output[-1].max(1)[1] == y).float().sum()

        if self.verbose:
            print('clean accuracy: {:.2%}'.format(acc / x_orig.shape[0]))

        return acc.item() / x_orig.shape[0]

    def run_standard_evaluation_individual(self, x_orig, y_orig, bs=250):
        if self.verbose:
            print('using {} version including {}'.format(self.version,
                                                         ', '.join(self.attacks_to_run)))

        l_attacks = self.attacks_to_run
        adv = {}
        verbose_indiv = self.verbose
        self.verbose = False

        for c in l_attacks:
            startt = time.time()
            self.attacks_to_run = [c]
            adv[c] = self.run_standard_evaluation(x_orig, y_orig, bs=bs)
            if verbose_indiv:
                acc_indiv = self.clean_accuracy(adv[c], y_orig, bs=bs)
                space = '\t \t' if c == 'fab' else '\t'
                self.logger.log(
                    'float_dis by {} robust accuracy by {} {} {:.2%} \t (time attack: {:.1f} s)'.format(self.float_dis,
                                                                                                        c.upper(),
                                                                                                        space,
                                                                                                        acc_indiv,
                                                                                                        time.time() - startt))

        return adv

    def set_version(self, version='standard'):
        if version == 'standard-pre':
            self.mora.scale = self.scale
            self.attacks_to_run = ['mora-ce-5', 'mora-ce-0.0', 'mora-ce-2.5', 'mora-ce-7.5', 'mora-ce-10' ]
            self.mora.n_restarts = 1
            self.mora_targeted.n_restarts = 1
            self.mora_targeted.n_target_classes = 9

        elif version == 'standard-t-pre':
            if self.ensemble_pattern == 'softmax' or self.ensemble_pattern == 'voting':
                self.attacks_to_run = ['mora-ce-5', 'mora-ce-0.0', 'mora-ce-2.5', 'mora-ce-7.5', 'mora-ce-10',
                                       'mora_ce_t-5']
            elif self.ensemble_pattern == 'logits':
                self.attacks_to_run = ['mora-ce-5', 'mora-ce-0.0', 'mora-ce-2.5', 'mora-ce-7.5', 'mora-ce-10',
                                       'mora_ce_t-0']

            self.mora_targeted.scale = self.scale
            self.mora.n_restarts = 1
            self.mora_targeted.n_restarts = 1
            self.mora_targeted.n_target_classes = 9

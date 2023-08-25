import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.device = 'cuda'
        assert len(self.models) > 0

    def get_output_scale(self, output, value=10):
        std_max_out = []
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [item / value for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device).float()
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def forward(self, x):
        outputs_old = 0
        outputs_scale = 0
        outputs = []

        for i, model in enumerate(self.models):
            sub_model_output_original = model(x)
            outputs.append(sub_model_output_original)

        for i in range(len(outputs)):
            sub_model_output_original = outputs[i]
            sub_model_output_original_scale_10 = self.get_output_scale(sub_model_output_original.clone().detach(), 10)
            sub_model_output_original_scale_3 = self.get_output_scale(sub_model_output_original.clone().detach(), 1)
            outputs_old += F.softmax(sub_model_output_original / sub_model_output_original_scale_10, dim=-1)
            outputs_scale += F.softmax(sub_model_output_original / sub_model_output_original_scale_3, dim=-1)

        output_old = outputs_old / len(self.models)
        output_scale = outputs_scale / len(self.models)

        #  # only using this operation when ensemble prediction probability
        output_old = torch.clamp(output_old, min=1e-80)
        output_old = torch.log(output_old)

        output_scale = torch.clamp(output_scale, min=1e-80)
        output_scale = torch.log(output_scale)

        outputs.append(output_scale)
        outputs.append(output_old)
        return outputs

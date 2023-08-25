import torch
import torch.nn as nn
import torch.nn.functional as F


class Ensemble(nn.Module):
    def __init__(self, models):
        super(Ensemble, self).__init__()
        self.models = models
        self.device = 'cuda'
        assert len(self.models) > 0

    def get_output_scale(self, output):
        std_max_out = []
        maxk = max((10,))
        pred_val_out, pred_id_out = output.topk(maxk, 1, True, True)
        std_max_out.extend((pred_val_out[:, 0] - pred_val_out[:, 1]).cpu().numpy())
        scale_list = [item / 10.0 for item in std_max_out]
        scale_list = torch.tensor(scale_list).to(self.device).float()
        scale_list = torch.unsqueeze(scale_list, -1)
        return scale_list

    def forward(self, x):
        outputs_old = 0
        outputs = []
        for i, model in enumerate(self.models):
            sub_model_output_original = model(x)
            outputs.append(sub_model_output_original)
            outputs_old += sub_model_output_original
        output_old = outputs_old / len(self.models)
        outputs.append(output_old)
        outputs.append(output_old)
        return outputs


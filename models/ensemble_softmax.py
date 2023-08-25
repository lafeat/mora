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

    # def forward(self, x):
    #     if len(self.models) > 1:
    #         outputs_old = 0
    #         outputs_scale = 0
    #         outputs = []
    #         for model in self.models:
    #             sub_model_output_original = model(x)
    #             outputs.append(sub_model_output_original)
    #             sub_model_output_original_scale = self.get_output_scale(sub_model_output_original.clone().detach())
    #             # outputs.append(sub_model_output_original/sub_model_output_original_scale)
    #
    #             # ############ # ensemble prediction probability
    #             outputs_old += F.softmax(sub_model_output_original.clone(), dim=-1)
    #
    #             # ############ # # ensemle max
    #             # sub_model_output_original_scale = self.get_output_scale(sub_model_output_original.clone().detach())
    #             outputs_scale += F.softmax(sub_model_output_original/sub_model_output_original_scale, dim=-1)
    #             # outputs_scale += F.softmax(sub_model_output_original.clone(), dim=-1)
    #
    #             # ############ # # ensemble logits
    #             # outputs_old += sub_model_output_original
    #
    #         output_old = outputs_old / len(self.models)
    #         output_scale = outputs_scale / len(self.models)
    #
    #         #  only using this operation when ensemble prediction probability
    #         # output_old = torch.clamp(output_old, min=1e-40)
    #         # output_old = torch.log(output_old)
    #         outputs.append(output_scale)
    #         outputs.append(output_old)
    #         return outputs
    #     else:
    #         return self.models[0](x)

    def forward(self, x):
        outputs_old = 0
        # outputs_new = 0
        outputs_scale = 0
        outputs_logits = 0
        outputs = []
        for i, model in enumerate(self.models):
            sub_model_output_original = model(x)
            outputs.append(sub_model_output_original)
            sub_model_output_original_scale = self.get_output_scale(sub_model_output_original.clone().detach())
            # outputs.append(sub_model_output_original/sub_model_output_original_scale)

            # ############ # ensemble prediction probability
            outputs_old += F.softmax(sub_model_output_original, dim=-1)
            # outputs_new += sub_model_output_original
            # ############ # # ensemle max
            # sub_model_output_original_scale = self.get_output_scale(sub_model_output_original.clone().detach())
            outputs_scale += F.softmax(sub_model_output_original, dim=-1)
            outputs_logits += sub_model_output_original
            # outputs_scale += F.softmax(sub_model_output_original, dim=-1)

            # ############ # # ensemble logits
            # outputs_old += sub_model_output_original

        output_old = outputs_old / len(self.models)
        # output_new = outputs_new / len(self.models)
        output_scale = outputs_scale / len(self.models)
        output_logits = outputs_logits/len(self.models)

        #  # only using this operation when ensemble prediction probability
        output_old = torch.clamp(output_old, min=1e-80)
        output_old = torch.log(output_old)

        output_scale = torch.clamp(output_scale, min=1e-80)
        output_scale = torch.log(output_scale)

        # outputs.append(output_new)
        # outputs.append(output_logits)
        outputs.append(output_scale)
        # outputs.append(output_old)
        outputs.append(output_old)
        return outputs

    # def forward(self, x):
    #     output = []
    #     outputs_old = 0
    #     for i, model in enumerate(self.models):
    #         sub_model_output_original = model(x)
    #         output.append(sub_model_output_original)
    #         outputs_old += F.softmax(sub_model_output_original, dim=-1)
    #
    #     output_old = outputs_old / len(self.models)
    #     output_old = torch.clamp(output_old, min=1e-40)
    #     output_old = torch.log(output_old)
    #     output.append(output_old)
    #     return output

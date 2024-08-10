import torch
import torch.nn as nn
import convnext
import torch.nn.functional as F


class LowRankLayer(nn.Module):
    def __init__(self, linear, rank, alpha, use_dora=True):
        super().__init__()
        # rank: controls the inner dimension of the matrices A and B; controls the number of additional param
        # a key factor in determining the balance between model adaptability and parameter efficiency.
        # alpha: a scaling hyper-parameter applied to the output of the low-rank adaptation,
        # controls the extent to which the adapted layer's output is allowed to influence the original output

        self.use_dora = use_dora
        self.rank = rank # low-rank
        self.alpha = alpha # scaling hyper-parameter
        self.linear = linear
        self.in_dim = linear.in_features
        self.out_dim = linear.out_features

        # weights
        std_dev = 1 / torch.sqrt(torch.tensor(self.rank).float())
        self.A = nn.Parameter(torch.randn(self.in_dim, self.rank) * std_dev)
        self.B = nn.Parameter(torch.zeros(self.rank, self.out_dim))

        if self.use_dora:
            self.m = nn.Parameter(
            self.linear.weight.norm(p=2, dim=0, keepdim=True))
        else:
            self.m = None

    def forward(self, x):
        lora = self.A @ self.B # combine LoRA matrices
        if self.use_dora:
            numerator = self.linear.weight + self.alpha * lora.T
            denominator = numerator.norm(p=2, dim=0, keepdim=True)
            directional_component = numerator / denominator
            new_weight = self.m * directional_component
            return F.linear(x, new_weight, self.linear.bias)
        else:
            # combine LoRA with orig. weights
            combined_weight = self.linear.weight + self.alpha * lora.T
            return F.linear(x, combined_weight, self.linear.bias)


def get_model():
    model = convnext.convnext_tiny(pretrained=True, in_22k=False).requires_grad_(False)
    
    # dora
    rank = 4
    alpha = 8

    # fine tune last stage
    stage = model.stages[3]
    stage.requires_grad_(True)
    for block in stage:
        block.pwconv1 = LowRankLayer(block.pwconv1, rank, alpha, use_dora=True)
        block.pwconv2 = LowRankLayer(block.pwconv2, rank, alpha, use_dora=True)

    model.norm.requires_grad_(True)

    model.head = nn.Linear(in_features=768, out_features=6, bias=True)


    return model


def print_model_size(model):
    num_trainable_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("num trainable weights: ", num_trainable_params)

    # calculate the model size on disk
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    print(f"model size: {size_all_mb:.2f} MB")

# Copyright (c) 2019 Shigeki Karita  
#               2020 Mobvoi Inc (Binbin Zhang)  
#               2024 修改以支持NVIDIA TransformerEngine FP8  
#  
# Licensed under the Apache License, Version 2.0 (the "License");  
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at  
#  
#   http://www.apache.org/licenses/LICENSE-2.0  
#  
# Unless required by applicable law or agreed to in writing, software  
# distributed under the License is distributed on an "AS IS" BASIS,  
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and  
# limitations under the License.  
"""Positionwise feed forward layer definition."""  
  
import torch  
import transformer_engine.pytorch as te  
from transformer_engine.common import recipe  
  
  
class PositionwiseFeedForward(torch.nn.Module):  
    """Positionwise feed forward layer.  
  
    FeedForward are appied on each position of the sequence.  
    The output dim is same with the input dim.  
  
    Args:  
        idim (int): Input dimenstion.  
        hidden_units (int): The number of hidden units.  
        dropout_rate (float): Dropout rate.  
        activation (torch.nn.Module): Activation function  
    """  
  
    def __init__(  
            self,  
            idim: int,  
            hidden_units: int,  
            dropout_rate: float,  
            activation: torch.nn.Module = torch.nn.ReLU(),  
    ):  
        """Construct a PositionwiseFeedForward object."""  
        super(PositionwiseFeedForward, self).__init__()  
          
        # 使用TransformerEngine的Linear层替代普通Linear层  
        self.w_1 = te.Linear(idim, hidden_units)  
        self.activation = activation  
        self.dropout = torch.nn.Dropout(dropout_rate)  
        self.w_2 = te.Linear(hidden_units, idim)  
          
        # 创建FP8配方，用于FP8自动转换  
        self.fp8_recipe = recipe.DelayedScaling(  
            margin=0,  
            fp8_format=recipe.Format.E4M3  
        )  
  
    def forward(self, xs: torch.Tensor) -> torch.Tensor:  
        """Forward function.  
  
        Args:  
            xs: input tensor (B, L, D)  
        Returns:  
            output tensor, (B, L, D)  
        """  
        # 使用FP8自动转换包装计算过程  
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):  
            return self.w_2(self.dropout(self.activation(self.w_1(xs))))  
  
  
class MoEFFNLayer(torch.nn.Module):  
    """  
    Mixture of expert with Positionwise feed forward layer  
    See also figure 1 in https://arxiv.org/pdf/2305.15663.pdf  
    The output dim is same with the input dim.  
  
    Modified from https://github.com/Lightning-AI/lit-gpt/pull/823  
                  https://github.com/mistralai/mistral-src/blob/b46d6/moe_one_file_ref.py#L203-L219  
    Args:  
        n_expert: number of expert.  
        n_expert_per_token: The actual number of experts used for each frame  
        idim (int): Input dimenstion.  
        hidden_units (int): The number of hidden units.  
        dropout_rate (float): Dropout rate.  
        activation (torch.nn.Module): Activation function  
    """  
  
    def __init__(  
            self,  
            n_expert: int,  
            n_expert_per_token: int,  
            idim: int,  
            hidden_units: int,  
            dropout_rate: float,  
            activation: torch.nn.Module = torch.nn.ReLU(),  
    ):  
        super(MoEFFNLayer, self).__init__()  
        # 使用TransformerEngine的Linear层替代普通Linear层  
        self.gate = te.Linear(idim, n_expert, bias=False)  
        # 使用改造后的PositionwiseFeedForward作为专家  
        self.experts = torch.nn.ModuleList(  
            PositionwiseFeedForward(idim, hidden_units, dropout_rate,  
                                    activation) for _ in range(n_expert))  
        self.n_expert_per_token = n_expert_per_token  
          
        # 创建FP8配方，用于FP8自动转换  
        self.fp8_recipe = recipe.DelayedScaling(  
            margin=0,  
            fp8_format=recipe.Format.E4M3  
        )  
  
    def forward(self, xs: torch.Tensor) -> torch.Tensor:  
        """Foward function.  
        Args:  
            xs: input tensor (B, L, D)  
        Returns:  
            output tensor, (B, L, D)  
  
        """  
        # 使用FP8自动转换包装计算过程  
        with te.fp8_autocast(enabled=True, fp8_recipe=self.fp8_recipe):  
            B, L, D = xs.size()  # batch size, sequence length, embedding dimension (idim)  
            xs = xs.view(-1, D)  # (B*L, D)  
            router = self.gate(xs)  # (B*L, n_expert)  
            logits, indices = torch.topk(  
                router, self.n_expert_per_token  
            )  # probs:(B*L, n_expert), indices: (B*L, n_expert)  
            weights = torch.nn.functional.softmax(  
                logits, dim=1,  
                dtype=torch.float).to(dtype=xs.dtype)  # (B*L, n_expert_per_token)  
            output = torch.zeros_like(xs)  # (B*L, D)  
            for i, expert in enumerate(self.experts):  
                mask = indices == i  
                batch_idx, ith_expert = torch.where(mask)  
                output[batch_idx] += weights[batch_idx, ith_expert, None] * expert(  
                    xs[batch_idx])  
            return output.view(B, L, D)

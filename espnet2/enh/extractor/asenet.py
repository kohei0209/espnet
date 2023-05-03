from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder

from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import TemporalConvNet, TemporalConvNetInformed
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ASENet(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        separator_hidden_dim: int = 512,
        separator_num_layers: int = 1,
        maskestimator_hidden_dim: int = 512,
        maskestimator_num_layers: int = 2,
        attention_dim: int = 200,
        dropout_p: float = 0.3,
        predict_noise: bool = False,
    ):
        """Attention-based speech separation and extraction network (ASENet).

        Args:
            input_dim: input feature dimension
            layer: int, number of layers in each stack
            stack: int, number of stacks
            bottleneck_dim: bottleneck dimension
            hidden_dim: number of convolution channel
            skip_dim: int, number of skip connection channels
            kernel: int, kernel size.
            causal: bool, defalut False.
            norm_type: str, choose from 'BN', 'gLN', 'cLN'
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            i_adapt_layer: int, index of adaptation layer
            adapt_layer_type: str, type of adaptation layer
                see espnet2.enh.layers.adapt_layers for options
            adapt_enroll_dim: int, dimensionality of the speaker embedding
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise

        # separator modules
        self.separator = nn.LSTM(
            input_dim,
            separator_hidden_dim,
            num_layers=separator_num_layers,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.separator_linear1 = nn.Linear(separator_hidden_dim*2, maskestimator_hidden_dim)
        self.separator_linear2 = nn.Linear(separator_hidden_dim*2, maskestimator_hidden_dim)

        # attention modules
        self.mlp1 = nn.Sequential(
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.ReLU(),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.ReLU(),
        )

        self.mlp_aux = nn.Sequential(
            nn.Linear(input_dim, maskestimator_hidden_dim),
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.ReLU(),
        )

        self.W_v = nn.Linear(maskestimator_hidden_dim, attention_dim)
        self.W_iv = nn.Linear(maskestimator_hidden_dim, attention_dim)
        self.W_aux = nn.Linear(maskestimator_hidden_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1)

        self.attention_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-3)
        self.alpha = 2

        # mask estimator modules
        self.mask_estimator_lstm = nn.LSTM(
                maskestimator_hidden_dim,
                maskestimator_hidden_dim,
                num_layers=maskestimator_num_layers,
                dropout=dropout_p,
                bidirectional=True,
            )
        self.mask_estimator_output = nn.Sequential(
            nn.Linear(maskestimator_hidden_dim*2, input_dim),
            nn.Sigmoid(),
        )


    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        input_aux: torch.Tensor = None,
        ilens_aux: torch.Tensor = None,
        suffix_tag: str = "",
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """TD-SpeakerBeam Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            input_aux (torch.Tensor or ComplexTensor): Encoded auxiliary feature
                for the target speaker [B, T, N]
            ilens_aux (torch.Tensor): input lengths of auxiliary input for the
                target speaker [Batch]
            suffix_tag (str): suffix to append to the keys in `others`

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                f'mask{suffix_tag}': torch.Tensor(Batch, Frames, Freq),
                f'enroll_emb{suffix_tag}': torch.Tensor(Batch, adapt_enroll_dim/adapt_enroll_dim*2),
            ]
        """  # noqa: E501

        is_tse = isinstance(input_aux, torch.Tensor)

        feature = abs(input)

        # separator part
        feature, _ = self.separator(feature)
        Z1 = self.separator_linear1(feature)
        Z2 = self.separator_linear2(feature)
        Z = torch.stack((Z1, Z2), dim=1) # [B, I, T, F]

        # attention part
        s_v = self.mlp1(Z) # [B, I, T, F]
        s_v2 = self.mlp2(Z).mean(dim=-2, keepdim=True) # [B, I, 1, F]

        # if 
        if is_tse:
            # aux_batch = self.enc(input_aux, ilens_aux)[0]  # [B, S, T, F]
            # aux_feature = aux_batch.transpose(1, 2) # [B, T, F]
            aux_feature = abs(input_aux)
            s_aux = self.mlp_aux(aux_feature).mean(dim=-2, keepdim=True) # [B, S, 1, F]

            # e: [B, S, I, T, F], a: [B, S, I, T, 1]
            e1 = self.W_v(s_v)
            e2 = self.W_iv(s_v2)
            e3 = self.W_aux(s_aux)
            e = e1[..., None, :, :, :] + e2[..., None, :, :, :] + e3[..., None, None, :]
            # e = self.W_v(s_v)[..., None, :, :, :] + self.W_iv(s_v2)[..., None, :, :, :] + self.W_aux(s_aux)[..., None, :, :]
            e = self.w(self.attention_activation(e))
            a = self.softmax(e/self.alpha)

            others = {"enroll_emb{}".format(suffix_tag): s_aux.detach(),}
        else:
            # force the attention weights to have 0 and 1 for each speaker
            B, I, T, F = Z.shape
            zeros, ones = Z.new_zeros((B, 1, 1, T, 1)), Z.new_ones((B, 1, 1, T, 1))
            a = torch.cat((torch.cat((ones, zeros), dim=2), torch.cat((zeros, ones), dim=2)), dim=1)
            others = {}
        # attention
        z_att = (a * Z[..., None, :, : ,:]).sum(dim=-3) # [B, S, T, F]
        B, S, T, F = z_att.shape
        z_att = z_att.reshape(-1, T, F)
        # mask estimation part
        masks, _ = self.mask_estimator_lstm(z_att) # [B*S, T, F]
        masks = self.mask_estimator_output(masks)
        masks = masks.reshape(B, S, T, -1).unbind(dim=-3)

        # masking
        if self.predict_noise:
            *masks, mask_noise = masks
        masked = [input * m for m in masks]
        if is_tse:
            masked = masked[0]

        others = OrderedDict(
            zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        )
        if self.predict_noise:
            others["noise1"] = input * mask_noise
        if isinstance(input_aux, torch.Tensor):
            others["enroll_emb{}".format(suffix_tag)] = s_aux.detach()
        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

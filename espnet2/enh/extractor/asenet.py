from collections import OrderedDict
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

from espnet2.enh.decoder.stft_decoder import STFTDecoder
from espnet2.enh.encoder.stft_encoder import STFTEncoder

from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import TemporalConvNet, TemporalConvNetInformed
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class ASENet(AbsExtractor):
    def __init__(
        self,
        input_dim: int,
        n_fft=128,
        stride=64,
        window="hann",
        separator_hidden_dim: int = 512,
        separator_num_layers: int = 1,
        maskestimator_hidden_dim: int = 512,
        maskestimator_num_layers: int = 2,
        dropout_p: float = 0.3,
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

        self.enc = STFTEncoder(n_fft, n_fft, stride, window=window, use_builtin_complex=False)
        self.dec = STFTDecoder(n_fft, n_fft, stride, window=window)
        n_freqs = n_fft // 2 + 1

        # separator modules
        self.separator = nn.LSTM(
            n_freqs,
            separator_hidden_dim,
            num_layers=separator_num_layers,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.separator_linear1 = nn.Linear(separator_hidden_dim, maskestimator_hidden_dim)
        self.separator_linear2 = nn.Linear(separator_hidden_dim, maskestimator_hidden_dim)

        # attention modules
        self.mlp1 = nn.ModuleList([
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.ReLU(),
        ])

        self.mlp2 = nn.ModuleList([
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.ReLU(),
        ])

        self.mlp_aux = nn.ModuleList([
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.Linear(maskestimator_hidden_dim, maskestimator_hidden_dim),
            nn.ReLU(),
        ])

        self.W_v = self.linear(maskestimator_hidden_dim, maskestimator_hidden_dim)
        self.W_iv = self.linear(maskestimator_hidden_dim, maskestimator_hidden_dim)
        self.W_aux = self.linear(maskestimator_hidden_dim, maskestimator_hidden_dim)
        self.w = self.linear(maskestimator_hidden_dim, 1)

        self.attention_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-3)
        self.alpha = 2

        # mask estimator modules
        self.mask_estimator = nn.ModuleList([
            nn.LSTM(
                maskestimator_hidden_dim,
                maskestimator_hidden_dim,
                num_layers=maskestimator_num_layers,
                dropout=dropout_p,
                bidirectional=True,
            ),
            nn.Linear(maskestimator_hidden_dim, n_freqs),
            nn.Sigmoid()
        ])


    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        input_aux: torch.Tensor,
        ilens_aux: torch.Tensor,
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
        batch = self.enc(input, ilens)[0]  # [B, T, F]
        # feature = batch.transpose(1, 2) # [B, T, F]
        feature = abs(batch)

        # separator part
        feature = self.separator(feature)
        Z1 = self.separator_linear1(feature)
        Z2 = self.separator_linear2(feature)
        Z = torch.stack((Z1, Z2), dim=1) # [B, I, T, F]

        # attention part
        s_v = self.mlp1(Z) # [B, I, T, F]
        s_v2 = self.mlp2(Z).mean(dim=-2, keepdim=True) # [B, I, 1, F]

        if input_aux is not None:
            aux_batch = self.enc(input_aux, ilens_aux)[0]  # [B, S, T, F]
            # aux_feature = aux_batch.transpose(1, 2) # [B, T, F]
            aux_feature = abs(aux_batch)
            s_aux = self.mlp_aux(aux_feature).mean(dim=-2, leepdim=True) # [B, S, 1, F]

            # e: [B, S, I, T, F], a: [B, S, I, T, 1]
            e = self.W_v(s_v)[..., None, :, :, :] + self.W_iv(s_v2)[..., None, :, :, :] + self.W_aux(s_aux)[..., None, :, :]
            e = self.w(self.attention_activation(e))
            a = self.softmax(e/self.alpha)
        # to do: write "else" part

        z_att = (a * Z[..., None, :, : ,:]).sum(dim=-3) # [B, S, T, F]

        # mask estimation part
        masks = self.mask_estimator(z_att) # [B, S, T, F]
        assert masks.shape[-3] == 1  # currently assuming #spks=1
        masked = batch[..., None, :, :] * masks

        others = {
            "enroll_emb{}".format(suffix_tag): s_aux.detach(),
        }

        return masked, ilens, others

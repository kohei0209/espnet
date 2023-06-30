from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.tcn import choose_norm
from espnet2.enh.layers.dptnet import DPTNet
from espnet2.enh.layers.dptnet_eda import DPTNet_EDA_Informed
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class DPTNetEDAExtractor(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        post_enc_relu: bool = True,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        unit: int = 256,
        att_heads: int = 4,
        dropout: float = 0.0,
        activation: str = "relu",
        norm_type: str = "gLN",
        layer: int = 6,
        segment_size: int = 20,
        nonlinear: str = "relu",
        # eda realted arguments
        i_eda_layer: int = 1,
        num_eda_modules: int = 1,
        triple_path: bool = True,
        # enrollment related arguments
        i_adapt_layer: int = 1,
        adapt_layer_type: str = "mul",
        adapt_enroll_dim: int = 128,
        adapt_attention_dim: int = 512,
        adapt_hidden_dim: int = 512,
        adapt_softmax_temp: int = 2,
    ):
        """Dual-Path RNN (DPRNN) Separator

        Args:
            input_dim: input feature dimension
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            num_spk: number of speakers
            predict_noise: whether to output the estimated noise signal
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
            layer: int, number of stacked RNN layers. Default is 3.
            unit: int, dimension of the hidden state.
            segment_size: dual-path segment size
            dropout: float, dropout ratio. Default is 0.
        """
        super().__init__()

        self._num_spk = num_spk
        self.predict_noise = predict_noise
        self.segment_size = segment_size
        self.input_dim = input_dim

        self.post_enc_relu = post_enc_relu
        self.enc_LN = choose_norm(norm_type, input_dim)
        self.dptnet = DPTNet_EDA_Informed(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim,
            att_heads=att_heads,
            dropout=dropout,
            activation=activation,
            num_layers=layer,
            bidirectional=bidirectional,
            norm_type=norm_type,
            triple_path=triple_path,
            i_eda_layer=i_eda_layer,
            num_eda_modules=num_eda_modules,
            i_adapt_layer=i_adapt_layer,
            adapt_layer_type=adapt_layer_type,
            adapt_enroll_dim=adapt_enroll_dim,
            adapt_attention_dim=adapt_attention_dim,
            adapt_hidden_dim=adapt_hidden_dim,
            adapt_softmax_temp=adapt_softmax_temp,
        )

        # Auxiliary network
        self.auxiliary_net = DPTNet(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            output_size=input_dim,
            att_heads=att_heads,
            dropout=dropout,
            activation=activation,
            num_layers=2,
            bidirectional=bidirectional,
            norm_type=norm_type,
        )
        # gated output layer
        self.output = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, 1), torch.nn.Tanh()
        )
        self.output_gate = torch.nn.Sequential(
            torch.nn.Conv1d(input_dim, input_dim, 1), torch.nn.Sigmoid()
        )

        if nonlinear not in ("sigmoid", "relu", "prelu", "tanh"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))
        assert nonlinear != "relu", "Final ReLU activation leads to worse performance and should not be used"

        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "prelu": torch.nn.PReLU(),
            "tanh": torch.nn.Tanh(),
        }[nonlinear]

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        input_aux: torch.Tensor = None,
        ilens_aux: torch.Tensor = None,
        suffix_tag: str = "",
        num_spk: int = None,
    ) -> Tuple[List[Union[torch.Tensor, ComplexTensor]], torch.Tensor, OrderedDict]:
        """Forward.

        Args:
            input (torch.Tensor or ComplexTensor): Encoded feature [B, T, N]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, N), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """

        # process the input feature (mixture)
        # if complex spectrum,
        if is_complex(input):
            feature = abs(input)
        elif self.post_enc_relu:
            feature = torch.nn.functional.relu(input)
        else:
            feature = input
        B, T, N = feature.shape
        feature = feature.transpose(1, 2)  # B, N, T
        feature = self.enc_LN(feature)
        segmented = self.split_feature(feature)  # B, N, L, K

        is_tse = input_aux is not None
        if is_tse:
            aux_feature = abs(input_aux) if is_complex(input_aux) else input_aux
            aux_feature = aux_feature.transpose(1, 2)  # B, N, T
            aux_segmented = self.split_feature(aux_feature)  # B, N, L, K: batch, hidden_dim, segment_len, num_segments
            enroll_emb = self.auxiliary_net(aux_segmented)  # B, N, L, K
        else:
            enroll_emb = None

        # dual-path block
        # should be modified to receive the num_spk as the forward argument instead of using self.num_spk
        processed, probabilities = self.dptnet(segmented, enroll_emb, num_spk=num_spk)  # B, N, L, K
        processed = processed.reshape(
            -1, self.input_dim, processed.size(-2), processed.size(-1)
        )  # B*num_spk, N, L, K
        # overlap-add
        processed = self.merge_feature(processed, length=T)  # B*num_spk, N, T
        # gated output layer for filter generation (B*num_spk, N, T)
        processed = self.output(processed) * self.output_gate(processed)
        masks = processed.reshape(B, -1, N, T)

        # list[(B, T, N)]
        masks = self.nonlinear(masks.transpose(-1, -2)).unbind(dim=1)

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
        if probabilities is not None:
            others["existance_probability"] = probabilities

        return masked, ilens, others

    def split_feature(self, x):
        B, N, T = x.size()
        unfolded = torch.nn.functional.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.segment_size, 1),
            padding=(self.segment_size, 0),
            stride=(self.segment_size // 2, 1),
        )
        return unfolded.reshape(B, N, self.segment_size, -1)

    def merge_feature(self, x, length=None):
        B, N, L, n_chunks = x.size()
        hop_size = self.segment_size // 2
        if length is None:
            length = (n_chunks - 1) * hop_size + L
            padding = 0
        else:
            padding = (0, L)

        seq = x.reshape(B, N * L, n_chunks)
        x = torch.nn.functional.fold(
            seq,
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )
        norm_mat = torch.nn.functional.fold(
            input=torch.ones_like(seq),
            output_size=(1, length),
            kernel_size=(1, L),
            padding=padding,
            stride=(1, hop_size),
        )

        x /= norm_mat

        return x.reshape(B, N, length)

    @property
    def num_spk(self):
        return self._num_spk

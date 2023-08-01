from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
from packaging.version import parse as V
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.dprnn_eda import DPRNN, merge_feature, split_feature
from espnet2.enh.layers.dprnn_eda import DPRNN_EDA_Informed
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class DPRNNEDAExtractor(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        output_size:int = None,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_spk: int = 2,
        predict_noise: bool = False,
        nonlinear: str = "relu",
        layer: int = 6,
        unit: int = 128,
        segment_size: int = 100,
        dropout: float = 0.0,
        # eda realted arguments
        i_eda_layer: int = 1,
        # enrollment related arguments
        i_adapt_layer: int = 1,
        adapt_layer_type: str = "mul",
        adapt_enroll_dim: int = 128,
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
        self.num_outputs = self.num_spk + 1 if self.predict_noise else self.num_spk
        self.segment_size = segment_size

        if output_size is None:
            output_size = input_dim

        self.dprnn = DPRNN_EDA_Informed(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            # output_size=input_dim * self.num_outputs,
            output_size=output_size,
            segment_size=segment_size,
            dropout=dropout,
            num_layers=layer,
            bidirectional=bidirectional,
            i_eda_layer=i_eda_layer,
            i_adapt_layer=i_adapt_layer,
            adapt_layer_type=adapt_layer_type,
            adapt_enroll_dim=adapt_enroll_dim,
        )

        # Auxiliary network
        self.auxiliary_net = DPRNN(
            rnn_type=rnn_type,
            input_size=input_dim,
            hidden_size=unit,
            output_size=adapt_enroll_dim,
            dropout=dropout,
            num_layers=2,
            bidirectional=bidirectional,
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
        feature = abs(input) if is_complex(input) else input
        B, T, N = feature.shape
        feature = feature.transpose(1, 2)  # B, N, T
        segmented, rest = split_feature(
            feature, segment_size=self.segment_size
        )  # B, N, L, K: batch, hidden_dim, segment_len, num_segments

        is_tse = input_aux is not None
        if is_tse:
            aux_feature = abs(input_aux) if is_complex(input_aux) else input_aux
            aux_feature = aux_feature.transpose(1, 2)  # B, N, T
            aux_segmented, aux_rest = split_feature(
                aux_feature, segment_size=self.segment_size,
            )  # B, N, L, K: batch, hidden_dim, segment_len, num_segments
            enroll_emb = self.auxiliary_net(aux_segmented)  # B, N, L, K
            # enroll_emb = merge_feature(enroll_emb, aux_rest)
            # enroll_emb.masked_fill_(make_pad_mask(ilens_aux, enroll_emb, -1), 0.0)
            # enroll_emb = enroll_emb.mean(dim=-1)  # B, N, K
        else:
            enroll_emb = None

        # dual-path block
        # should be modified to receive the num_spk as the forward argument instead of using self.num_spk
        processed, probabilities = self.dprnn(segmented, enroll_emb, num_spk=num_spk)  # B, N, L, K
        # overlap-add
        processed = merge_feature(processed, rest)  # B, N*num_spk, T
        processed = processed.transpose(1, 2)  # B, T, N*num_spk
        # num_spk = 1 if is_tse else self.num_outputs
        processed = processed.view(B, T, N, -1)
        masks = self.nonlinear(processed).unbind(dim=3)

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

    @property
    def num_spk(self):
        return self._num_spk

from collections import OrderedDict
from distutils.version import LooseVersion
from typing import List
from typing import Tuple
from typing import Union

import torch
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.layers.tfpsnet import TFPSNet_Transformer_EDA, TFPSNet_Transformer
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.extractor.abs_extractor import AbsExtractor


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class TFPSNetEDAExtractor(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        enc_channels: int = 256,
        bottleneck_size: int = 64,
        separator_type: str = "transformer",
        tfps_blocks: list = [1, 2, 1, 1, 2, 1],
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        unit: int = 256,
        dropout: float = 0.0,
        norm_type: str = "gLN",
        nonlinear: str = "relu",
        masking: bool = True,
        # eda realted arguments
        i_eda_layer: int = 4,
        num_eda_modules: int = 1,
        # enrollment related arguments
        i_adapt_layer: int = 4,
        adapt_layer_type: str = "tfattn",
        adapt_enroll_dim: int = 64,
        adapt_attention_dim: int = 512,
        adapt_hidden_dim: int = 512,
        adapt_softmax_temp: int = 1,
    ):
        """Time-Frequency Domain Path Scanning Network (TFPSNet) Separator.

        Reference:
            [1] L. Yang, W. Liu, and W. Wang, “TFPSNet: Time-frequency domain
            path scanning network for speech separation,” in Proc. IEEE ICASSP,
            2022, pp. 6842-6846.

        Args:
            input_dim: int, input feature dimension.
            num_spk: int, number of speakers.
            enc_channels: int, feature dimension after the Conv1D encoder.
            bottleneck_size: int, dimension of the bottleneck feature.
            separator_type: string, select from "rnn" and "transformer".
            tfps_blocks: list, a series of integers (1 or 2) indicating the TFPSBlock type.
            rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
            bidirectional: bool, whether the inter-chunk RNN layers are bidirectional.
            unit: int, dimension of the hidden state.
            dropout: float, dropout ratio. Default is 0.
            norm_type: type of normalization to use after each inter- or
                intra-chunk NN block.
            nonlinear: the nonlinear function for mask estimation,
                       select from 'relu', 'tanh', 'sigmoid'
        """
        super().__init__()

        self._num_spk = num_spk

        self.enc_channels = enc_channels

        self.post_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(2, enc_channels, 6, bias=False),
            torch.nn.ReLU(),
        )

        assert separator_type in ("transformer"), separator_type
        if separator_type == "transformer":
            self.tfpsnet = TFPSNet_Transformer_EDA(
                enc_channels,
                bottleneck_size,
                output_size=enc_channels,
                tfps_blocks=tuple(tfps_blocks),
                # Transformer-specific arguments
                rnn_type=rnn_type,
                hidden_size=unit,
                att_heads=4,
                dropout=dropout,
                activation="relu",
                bidirectional=bidirectional,
                norm_type=norm_type,
                # EDA-related
                i_eda_layer=i_eda_layer,
                num_eda_modules=num_eda_modules,
                # TSE-related
                i_adapt_layer=i_adapt_layer,
                adapt_layer_type=adapt_layer_type,
                adapt_enroll_dim=adapt_enroll_dim,
                adapt_attention_dim=adapt_attention_dim,
                adapt_hidden_dim=adapt_hidden_dim,
                adapt_softmax_temp=adapt_softmax_temp,
            )
            # if i_adapt_layer is not None:
            #     self.auxiliary_net = TFPSNet_Transformer(
            #         enc_channels,
            #         bottleneck_size,
            #         output_size=enc_channels,
            #         tfps_blocks=(1,),
            #         # Transformer-specific arguments
            #         rnn_type=rnn_type,
            #         hidden_size=unit,
            #         att_heads=4,
            #         dropout=dropout,
            #         activation="relu",
            #         bidirectional=bidirectional,
            #         norm_type=norm_type,
            #     )

        # gated output layer
        if masking:
            self.output = torch.nn.Sequential(
                torch.nn.Conv1d(enc_channels, enc_channels, 1), torch.nn.Tanh()
            )
            self.output_gate = torch.nn.Sequential(
                torch.nn.Conv1d(enc_channels, enc_channels, 1), torch.nn.Sigmoid()
            )

        if nonlinear not in ("sigmoid", "relu", "tanh", "linear"):
            raise ValueError("Not supporting nonlinear={}".format(nonlinear))
        self.nonlinear = {
            "sigmoid": torch.nn.Sigmoid(),
            "relu": torch.nn.ReLU(),
            "tanh": torch.nn.Tanh(),
            "linear": torch.nn.Identity(),
        }[nonlinear]

        # self.pre_decoder = torch.nn.Linear(enc_channels, 2)
        self.pre_decoder = torch.nn.Conv1d(enc_channels, 2, 1)

        self.masking = masking

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
            input (torch.Tensor or ComplexTensor): STFT feature [B, T, F]
            ilens (torch.Tensor): input lengths [Batch]
            additional (Dict or None): other data included in model
                NOTE: not used in this model

        Returns:
            masked (List[Union(torch.Tensor, ComplexTensor)]): [(B, T, F), ...]
            ilens (torch.Tensor): (B,)
            others predicted data, e.g. masks: OrderedDict[
                'mask_spk1': torch.Tensor(Batch, Frames, Freq),
                'mask_spk2': torch.Tensor(Batch, Frames, Freq),
                ...
                'mask_spkn': torch.Tensor(Batch, Frames, Freq),
            ]
        """
        # B, T, 2F
        if is_complex(input):
            feature = torch.stack([input.real, input.imag], dim=-2)  # B, T, 2, F
            feature = feature.moveaxis(1, -1)
        else:
            # real and imag parts
            assert input.size(-1) == 2, input.shape
            feature = input.permute(0, 3, 2, 1)
            feature = ComplexTensor([..., 0], input[..., 1])

        B, _, F, T = feature.shape

        feature = feature.moveaxis(-1, 1).reshape(B * T, 2, F)
        feature = torch.nn.functional.pad(feature, (0, 5), mode="circular")

        feature = self.post_encoder(feature)  # B*T, enc_channels, F

        feature = feature.reshape(B, T, -1, F).moveaxis(1, -1)  # B, enc_channels, F, T

        is_tse = input_aux is not None
        if is_tse:
            if is_complex(input_aux):
                enroll_emb = torch.stack([input_aux.real, input_aux.imag], dim=-2)  # B, T, 2, F
                enroll_emb = enroll_emb.moveaxis(1, -1)
                T_emb = enroll_emb.shape[-1]
                enroll_emb = enroll_emb.moveaxis(-1, 1).reshape(B * T_emb, 2, F)
                enroll_emb = torch.nn.functional.pad(enroll_emb, (0, 5), mode="circular")
                enroll_emb = self.post_encoder(enroll_emb)  # B*T, enc_channels, F
                enroll_emb = enroll_emb.reshape(B, T_emb, -1, F).moveaxis(1, -1)  # B, enc_channels, F, T
                # enroll_emb = self.auxiliary_net(enroll_emb)
            else:
                # real and imag parts
                raise NotImplementedError("Input must be complex")
        else:
            enroll_emb = None

        processed, probabilities = self.tfpsnet(feature, enroll_emb, num_spk=num_spk)  # B, enc_channels*num_spk, F, T
        if self.masking:
            processed = processed.reshape(-1, self.enc_channels, F * T)
            # gated output layer for filter generation (B*num_spk, enc_channels, F*T)
            masks = self.output(processed) * self.output_gate(processed)
            masks = self.nonlinear(masks.reshape(B, -1, self.enc_channels, F, T))
            masked = feature.unsqueeze(1) * masks
        else:
            masked = processed.reshape(-1, self.enc_channels, F, T)
        masked = masked.moveaxis(-1, 2).reshape(
            -1, self.enc_channels, F
        )
        masked = (
            self.pre_decoder(masked).reshape(B, -1, T, 2, F).moveaxis(-1, -2)
        )

        # B, num_spk, T, F, enc_channels
        # masked = self.pre_decoder(masked.permute(0, 1, 4, 3, 2))
        masked = new_complex_like(input, (masked[..., 0], masked[..., 1])).unbind(1)
        if is_tse:
            masked = masked[0]
        if self.masking:
            others = OrderedDict(
                zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
            )
        else:
            others = {}
        if probabilities is not None:
            others["existance_probability"] = probabilities
        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk

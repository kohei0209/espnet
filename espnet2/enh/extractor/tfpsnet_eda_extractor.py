from collections import OrderedDict
from distutils.version import LooseVersion
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.layers.complex_utils import new_complex_like
from espnet2.enh.layers.tfpsnet import TFPSNet_Transformer_EDA, TFPSBlockType1, Conditional_TFPSNet_Transformer
from espnet2.enh.separator.abs_separator import AbsSeparator
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.layers.dptnet import (
    ImprovedTransformerLayer as SingleTransformer,
)
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")


class TFPSNetEDAExtractor(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        enc_channels: int = 256,
        bottleneck_size: int = 64,
        separator_type: str = "transformer",
        tfps_blocks: list = [1, 1, 1, 1, 1, 1],
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
        unit_after_eda: int = 256,
        sep_algo: str = "multiply",
        checkpointing: bool = False,
        # enrollment related arguments
        i_adapt_layer: int = 4,
        num_aux_tfps_blocks: int = 1,
        cond_tfps_blocks: list = None,
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
                hidden_size_after_eda=unit_after_eda,
                sep_algo=sep_algo,
                checkpointing=checkpointing,
                # TSE-related
                i_adapt_layer=i_adapt_layer,
                num_aux_tfps_blocks=num_aux_tfps_blocks,
                cond_tfps_blocks=cond_tfps_blocks,
                adapt_layer_type=adapt_layer_type,
                adapt_enroll_dim=adapt_enroll_dim,
                adapt_attention_dim=adapt_attention_dim,
                adapt_hidden_dim=adapt_hidden_dim,
                adapt_softmax_temp=adapt_softmax_temp,
            )
            if i_adapt_layer is not None:
                self.post_encoder_enroll = torch.nn.Sequential(
                    torch.nn.Conv1d(2, enc_channels, 6, bias=False),
                    torch.nn.ReLU(),
                )

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
        task: str = None,
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

        # is_tse = input_aux is not None
        if task is None and input_aux is None:
            task = "enh"
        elif task is None and input_aux is not None:
            task = "tse"
        assert task in ["enh", "tse", "enh_tse"]

        if "tse" in task:
            if is_complex(input_aux):
                enroll_emb = torch.stack([input_aux.real, input_aux.imag], dim=-2)  # B, T, 2, F
                enroll_emb = enroll_emb.moveaxis(1, -1)
                T_emb = enroll_emb.shape[-1]
                enroll_emb = enroll_emb.moveaxis(-1, 1).reshape(B * T_emb, 2, F)
                enroll_emb = torch.nn.functional.pad(enroll_emb, (0, 5), mode="circular")
                enroll_emb = self.post_encoder_enroll(enroll_emb)  # B*T, enc_channels, F
                enroll_emb = enroll_emb.reshape(B, T_emb, -1, F).moveaxis(1, -1)  # B, enc_channels, F, T
                # enroll_emb = self.auxiliary_net(enroll_emb)
            else:
                # real and imag parts
                raise NotImplementedError("Input must be complex")
        else:
            enroll_emb = None

        processed, probabilities = self.tfpsnet(feature, enroll_emb, num_spk=num_spk, task=task)  # B, enc_channels*num_spk, F, T
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
        if task == "tse":
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


class MultiDecoderTFPSNetExtractor(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk_list: List,
        num_spk: int = 2,
        enc_channels: int = 256,
        bottleneck_size: int = 64,
        separator_type: str = "transformer",
        tfps_blocks: list = [1, 1, 1, 1, 1, 1],
        att_heads: int = 4,
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        unit: int = 256,
        dropout: float = 0.0,
        norm_type: str = "gLN",
        nonlinear: str = "relu",
        masking: bool = True,
        # mutli-decoder realted arguments
        num_decoder_layer: int = 2,
        decoder_unit: int = 256,
        checkpointing: bool = False,
        # enrollment related arguments
        use_tse_decoder: bool = False,
        num_aux_tfps_blocks: int = 2,
        adapt_hidden_dim: int = 512,
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

        # encoder related
        self.enc_channels = enc_channels
        # [B, enc_channels, T] -> [B, .enc_channels, T]
        self.layer_norm = ChannelwiseLayerNorm(enc_channels)
        # [B, self.enc_channels, T] -> [B, bottleneck_size, T]
        self.bottleneck_conv1x1 = nn.Conv1d(
            enc_channels, bottleneck_size, 1, bias=False
        )
        self.post_encoder = torch.nn.Sequential(
            torch.nn.Conv1d(2, enc_channels, 6, bias=False),
            torch.nn.ReLU(),
        )

        self.bottleneck = nn.ModuleList()
        for block in tfps_blocks[:-num_decoder_layer]:
            assert block == 1, "Use Type1 block"
            self.bottleneck.append(
                TFPSBlockType1(
                    SingleTransformer,
                    rnn_type,
                    bottleneck_size,
                    att_heads,
                    unit,
                    dropout=dropout,
                    activation="relu",
                    bidirectional=bidirectional,
                    norm=norm_type,
                )
            )

        # define multi-decoders
        self.num_spk_list = num_spk_list
        self.num_decoder = num_decoder = len(num_spk_list)

        self.num_decoder_layer = num_decoder_layer
        self.masking = masking
        if num_decoder_layer > 0:
            self.decoder = nn.ModuleList()
        if masking:
            self.output_tanh = nn.ModuleList()
            self.output_sigmoid = nn.ModuleList()
            if nonlinear not in ("sigmoid", "relu", "tanh", "linear"):
                raise ValueError("Not supporting nonlinear={}".format(nonlinear))
            self.nonlinear = {
                "sigmoid": torch.nn.Sigmoid(),
                "relu": torch.nn.ReLU(),
                "tanh": torch.nn.Tanh(),
                "linear": torch.nn.Identity(),
            }[nonlinear]
        self.pre_decoder = nn.ModuleList()
        self.output = nn.ModuleList()

        if decoder_unit is None:
            decoder_unit = unit

        for ndec in range(num_decoder):
            # tfps blocks
            if num_decoder_layer > 0:
                decoder = nn.ModuleList()
                for _ in range(num_decoder_layer):
                    decoder.append(
                        TFPSBlockType1(
                            SingleTransformer,
                            rnn_type,
                            bottleneck_size,
                            att_heads,
                            decoder_unit,
                            dropout=dropout,
                            activation="relu",
                            bidirectional=bidirectional,
                            norm=norm_type,
                        )
                    )
                self.decoder.append(decoder)
            # output layer
            self.output.append(
                nn.Sequential(
                    nn.PReLU(),
                    nn.Conv2d(
                        bottleneck_size,
                        enc_channels * num_spk_list[ndec],
                        1
                    )
                )
            )
            # output gates for masking
            if masking:
                self.output_tanh.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(enc_channels, enc_channels, 1), torch.nn.Tanh(),
                    )
                )
                self.output_sigmoid.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(enc_channels, enc_channels, 1), torch.nn.Sigmoid(),
                    )
                )
            # used right before decoder
            self.pre_decoder.append(torch.nn.Conv1d(enc_channels, 2, 1))

        # speaker number estimator
        self.linear1 = nn.Sequential(
            nn.Linear(bottleneck_size, bottleneck_size),
            nn.PReLU(),
            nn.Linear(bottleneck_size, bottleneck_size),
        )
        self.linear2 = nn.Linear(bottleneck_size, num_decoder)
        self.softmax = nn.Softmax(dim=-1)

        # tse related
        # auxiliary network
        if use_tse_decoder:
            self.post_encoder_enroll = torch.nn.Sequential(
                torch.nn.Conv1d(2, enc_channels, 6, bias=False),
                torch.nn.ReLU(),
            )
            self.layer_norm_enroll = ChannelwiseLayerNorm(enc_channels)
            self.bottleneck_conv1x1_enroll = nn.Conv1d(
                enc_channels, bottleneck_size, 1, bias=False
            )
            self.auxiliary_net = nn.ModuleList()
            for i in range(num_aux_tfps_blocks):
                self.auxiliary_net.append(
                    TFPSBlockType1(
                        SingleTransformer,
                        rnn_type,
                        bottleneck_size,
                        att_heads,
                        unit,
                        dropout=dropout,
                        activation="relu",
                        bidirectional=bidirectional,
                        norm=norm_type,
                    )
                )
            # tse block
            self.tse_block = Conditional_TFPSNet_Transformer(
                bottleneck_size,
                adapt_hidden_dim,
                bottleneck_size,
                None,
                tfps_blocks[-num_decoder_layer:],
                SingleTransformer,
                rnn_type,
                bottleneck_size,
                att_heads,
                decoder_unit,
                dropout=dropout,
                activation="relu",
                bidirectional=bidirectional,
                norm=norm_type,
            )
            # output layer for tse
            self.output.append(
                nn.Sequential(
                    nn.PReLU(),
                    nn.Conv2d(
                        bottleneck_size,
                        enc_channels,
                        1
                    )
                )
            )
            # output gates for masking
            if masking:
                self.output_tanh.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(enc_channels, enc_channels, 1), torch.nn.Tanh(),
                    )
                )
                self.output_sigmoid.append(
                    torch.nn.Sequential(
                        torch.nn.Conv1d(enc_channels, enc_channels, 1), torch.nn.Sigmoid(),
                    )
                )
            # used right before decoder
            self.pre_decoder.append(torch.nn.Conv1d(enc_channels, 2, 1))

    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        input_aux: torch.Tensor = None,
        ilens_aux: torch.Tensor = None,
        suffix_tag: str = "",
        num_spk: int = None,
        task: str = None,
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

        # is_tse = input_aux is not None
        if task is None and input_aux is None:
            task = "enh"
        elif task is None and input_aux is not None:
            task = "tse"
        # assert not task == "enh_tse", "now we do not support enh_tse task"
        assert task in ["enh", "tse", "enh_tse"]

        if "tse" in task:
            if is_complex(input_aux):
                enroll_emb = torch.stack([input_aux.real, input_aux.imag], dim=-2)  # B, T, 2, F
                enroll_emb = enroll_emb.moveaxis(1, -1)
                T_emb = enroll_emb.shape[-1]
                enroll_emb = enroll_emb.moveaxis(-1, 1).reshape(B * T_emb, 2, F)
                enroll_emb = torch.nn.functional.pad(enroll_emb, (0, 5), mode="circular")
                enroll_emb = self.post_encoder_enroll(enroll_emb)  # B*T, enc_channels, F
                enroll_emb = enroll_emb.reshape(B, T_emb, -1, F).moveaxis(1, -1)  # B, enc_channels, F, T
            else:
                # real and imag parts
                raise NotImplementedError("Input must be complex")
        else:
            enroll_emb = None

        B, N, F, T = feature.shape
        output = self.layer_norm(feature.reshape(B, N, -1))
        output = self.bottleneck_conv1x1(output).reshape(
            B, -1, F, T
        )  # B, H, F, T
        for i, block in enumerate(self.bottleneck):
            assert isinstance(block, TFPSBlockType1), "Use TFPSBlockType1!"
            output = block(output)

        # estimate number of speakers
        # during training, probabilities are computed but oracle num_spk must be used
        probabilities = self.speaker_number_estimation(output)
        if num_spk is None:
            assert output.shape[0] == 1, "batchsize must be 1 during inference"
            idx = torch.argmax(probabilities, dim=-1)
            num_spk = self.num_spk_list[idx]
        else:
            idx = self.num_spk_list.index(num_spk)

        if "enh" in task:
            # speech separation
            # use corresponding decoder
            for i in range(self.num_decoder_layer):
                enh_output = self.decoder[idx][i](output)
            enh_output = self.after_decoder(enh_output, feature, idx)
        if "tse" in task:
            # target speaker extraction
            tse_output = self.tse_decoder(output, enroll_emb)
            tse_output = self.after_decoder(tse_output, feature, -1)

        # concat outputs when enh_tse task
        if task == "enh_tse":
            # concat enh output and tse output along n_src dim
            output = torch.cat((enh_output, tse_output), dim=1)
        elif task == "enh":
            output = enh_output
        else:
            output = tse_output

        '''
        # estimate multple speakers
        output = self.output[idx](output)
        output = output.reshape(-1, self.enc_channels, F * T)

        # masking
        if self.masking:
            masks = self.output_tanh[idx](output) * self.output_sigmoid[idx](output)
            masks = self.nonlinear(masks.reshape(B, -1, self.enc_channels, F, T))
            output = feature.unsqueeze(1) * masks
        else:
            output = output.reshape(-1, self.enc_channels, F, T)

        # final layer before istft
        output = output.moveaxis(-1, 2).reshape(
            -1, self.enc_channels, F
        )
        output = self.pre_decoder[idx](output).reshape(B, -1, T, 2, F).moveaxis(-1, -2)
        '''

        # B, num_spk, T, F, enc_channels
        output = new_complex_like(input, (output[..., 0], output[..., 1])).unbind(1)

        # if self.masking:
        #     others = OrderedDict(
        #         zip(["mask_spk{}".format(i + 1) for i in range(len(masks))], masks)
        #     )
        # else:
        #     others = {}
        others = {}
        if probabilities is not None:
            others["spkest_probability"] = probabilities
        return output, ilens, others

    def speaker_number_estimation(self, input, temp=1):
        # input is (B, H, F, T)
        probabilities = input.transpose(-1, -3)
        # mean along time and freq dim
        probabilities = self.linear1(probabilities).mean(dim=(-2, -3))
        probabilities = self.linear2(probabilities)
        probabilities = self.softmax(probabilities / temp)
        return probabilities

    def tse_decoder(self, input, enroll_emb):
        # get shapes
        B, N, F, T_enroll = enroll_emb.shape
        H = input.shape[-3]

        # process enroll embedding
        enroll_emb = self.layer_norm_enroll(enroll_emb.reshape(B, N, -1))
        enroll_emb = self.bottleneck_conv1x1_enroll(enroll_emb).reshape(
            B, H, F, T_enroll
        )
        for aux_block in self.auxiliary_net:
            enroll_emb = aux_block(enroll_emb)

        # extract only the target speaker
        output = input
        output = self.tse_block(output, enroll_emb.mean(dim=-1, keepdim=True))
        # for tse_block in self.tse_block:
        #     output = tse_block(output, enroll_emb)
        return output

    def after_decoder(self, input, feature, idx):
        B, _, F, T = input.shape
        # estimate multple speakers
        output = self.output[idx](input)
        output = output.reshape(-1, self.enc_channels, F * T)

        # masking
        if self.masking:
            masks = self.output_tanh[idx](output) * self.output_sigmoid[idx](output)
            masks = self.nonlinear(masks.reshape(B, -1, self.enc_channels, F, T))
            output = feature.unsqueeze(1) * masks
        else:
            output = output.reshape(-1, self.enc_channels, F, T)

        # final layer before istft
        output = output.moveaxis(-1, 2).reshape(
            -1, self.enc_channels, F
        )
        output = self.pre_decoder[idx](output).reshape(B, -1, T, 2, F).moveaxis(-1, -2)
        return output

    @property
    def num_spk(self):
        return self._num_spk

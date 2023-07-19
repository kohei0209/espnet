# The implementation of TFPSNet proposed in
# L. Yang, W. Liu, and W. Wang, “TFPSNet: Time-frequency domain path scanning network
# for speech separation,” in Proc. IEEE ICASSP, 2022, pp. 6842–6846.

import torch
import torch.nn as nn

from espnet2.enh.layers.dprnn import SingleRNN as SingleRNNLayer
from espnet2.enh.layers.dptnet import ImprovedTransformerLayer as SingleTransformer
from espnet2.enh.layers.tcn import ChannelwiseLayerNorm
from espnet2.enh.layers.tcn import choose_norm

from espnet2.enh.layers.adapt_layers import make_adapt_layer
from espnet2.enh.layers.dprnn_eda import SequenceAggregation, EncoderDecoderAttractor
from espnet2.enh.layers.dptnet_eda import FiLM


EPS = torch.finfo(torch.get_default_dtype()).eps


class SingleRNN(SingleRNNLayer):
    """Container module for a single RNN layer.

    args:
        rnn_type: string, select from 'RNN', 'LSTM' and 'GRU'.
        input_size: int, dimension of the input feature. The input should have shape
                    (batch, seq_len, input_size).
        hidden_size: int, dimension of the hidden state.
        dropout: float, dropout ratio. Default is 0.
        bidirectional: bool, whether the RNN layers are bidirectional. Default is False.
        norm (str, optional): Type of normalization to use.
    """

    def __init__(
        self,
        rnn_type,
        input_size,
        hidden_size,
        dropout=0,
        bidirectional=False,
        norm="gLN",
    ):
        super().__init__(
            rnn_type,
            input_size,
            hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.norm = choose_norm(norm, input_size)

    def forward(self, input):
        # input shape: batch, seq, dim
        # input = input.to(device)
        rnn_output, _ = self.rnn(input)
        rnn_output = self.dropout(rnn_output)
        rnn_output = self.proj(rnn_output.reshape(-1, rnn_output.shape[2])).view(
            input.shape
        )
        rnn_output = self.dropout(rnn_output) + rnn_output
        return self.norm(rnn_output.transpose(-1, -2)).transpose(-1, -2)


class TFPSNet_Base(nn.Module):
    """Base module for the Time-Frequency Domain Path Scanning Network.

    args:
        input_size (int): dimension of the input feature.
        bottleneck_size (int): dimension of the bottleneck feature.
        output_size (int): dimension of the output.
        tfps_blocks (tuple): a series of integers (1 or 2) indicating the TFPSBlock type.
        nn_module (torch.nn.Module): basic module for each path scanning
        *nn_args: positional arguments for initializing the NN module
        **nn_kwargs: keyword arguments for initializing the NN module
    """

    def __init__(
        self,
        input_size,
        bottleneck_size,
        output_size,
        tfps_blocks,
        nn_module,
        *nn_args,
        **nn_kwargs,
    ):
        super().__init__()

        # [B, input_size, T] -> [B, input_size, T]
        self.layer_norm = ChannelwiseLayerNorm(input_size)
        # [B, input_size, T] -> [B, bottleneck_size, T]
        self.bottleneck_conv1x1 = nn.Conv1d(input_size, bottleneck_size, 1, bias=False)

        self.input_size = input_size
        self.bottleneck_size = bottleneck_size
        self.output_size = output_size

        # dual-path transformer
        self.tfps_blocks = nn.ModuleList()
        for block in tfps_blocks:
            if block == 1:
                self.tfps_blocks.append(
                    TFPSBlockType1(nn_module, *nn_args, **nn_kwargs)
                )
            elif block == 2:
                self.tfps_blocks.append(
                    TFPSBlockType2(nn_module, *nn_args, **nn_kwargs)
                )
            else:
                raise ValueError(f"TFPSBlock type ({block}) must be either 1 or 2")

        # output layer
        self.output = nn.Sequential(
            nn.PReLU(), nn.Conv2d(bottleneck_size, output_size, 1)
        )

    def forward(self, input):
        B, N, F, T = input.shape
        output = self.layer_norm(input.reshape(B, N, -1))
        output = self.bottleneck_conv1x1(output).reshape(B, -1, F, T)  # B, BN, F, T
        for block in self.tfps_blocks:
            # print(f"output: {output.shape} -> ", end="")
            output = block(output)
            # print(output.shape)

        output = self.output(output)  # B, output_size, F, T
        return output


class TFPSBlockType1(nn.Module):
    def __init__(self, nn_module, *nn_args, **nn_kwargs):
        """Container module for a single TFPS Block (Type 1).

        It consists of a frequency scanning NN followed by a time scanning NN.

        Args:
            nn_module (torch.nn.Module): basic NN moudle
            nn_args (tuple): positional arguments for initializing the NN moudle
            nn_kwargs (dict): keyword arguments for initializing the NN moudle
        """
        super().__init__()

        self.fs_nn = nn_module(*nn_args, **nn_kwargs)
        self.ts_nn = nn_module(*nn_args, **nn_kwargs)

    def forward(self, input):
        """Forward.

        Args:
            input (torch.Tensor): feature sequence (batch, N, freq, time)

        Returns:
            output (torch.Tensor): output sequence (batch, N, freq, time)
        """
        output = self.freq_path_process(input)
        output = self.time_path_process(output)
        return output

    def freq_path_process(self, x):
        batch, N, freq, time = x.shape
        x = x.permute(0, 3, 2, 1).reshape(batch * time, freq, N)
        x = self.fs_nn(x)
        x = x.reshape(batch, time, freq, N).permute(0, 3, 2, 1)
        return x.contiguous()

    def time_path_process(self, x):
        batch, N, freq, time = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch * freq, time, N)
        x = self.ts_nn(x)
        x = x.reshape(batch, freq, time, N).permute(0, 3, 1, 2)
        return x.contiguous()


class TFPSBlockType2(nn.Module):
    def __init__(self, nn_module, *nn_args, **nn_kwargs):
        """Container module for a single TFPS Block (Type 2).

        It consists of a frequency scanning NN followed by a time-frequency scanning NN.

        Args:
            nn_module (torch.nn.Module): basic NN moudle
            nn_args (tuple): positional arguments for initializing the NN moudle
            nn_kwargs (dict): keyword arguments for initializing the NN moudle
        """
        super().__init__()

        self.fs_nn = nn_module(*nn_args, **nn_kwargs)
        self.tfs_nn = nn_module(*nn_args, **nn_kwargs)

    def forward(self, input, apply_attn_mask=False):
        """Forward.

        Args:
            input (torch.Tensor): feature sequence (batch, N, freq, time)
            apply_attn_mask (bool): whether to apply the attention mask
                for T-F path scanning

        Returns:
            output (torch.Tensor): output sequence (batch, N, freq, time)
        """
        output = self.freq_path_process(input)
        output = self.time_frequency_path_process(
            output, apply_attn_mask=apply_attn_mask
        )
        return output

    def freq_path_process(self, x):
        batch, N, freq, time = x.shape
        x = x.permute(0, 3, 2, 1).reshape(batch * time, freq, N)
        x = self.fs_nn(x)
        x = x.reshape(batch, time, freq, N).permute(0, 3, 2, 1)
        return x.contiguous()

    def time_frequency_path_process(self, x, apply_attn_mask=False):
        batch, N, freq, time = x.shape
        # batch * N, freq + time - 1, min(freq, time)
        antidiag_pad = self.get_right_padded_antidiagonals(x.reshape(-1, freq, time))
        _, num_antidiag, len_antidiag = antidiag_pad.shape
        x = (
            antidiag_pad.reshape(batch, N, *antidiag_pad.shape[-2:])
            .moveaxis(1, -1)
            .reshape(batch * num_antidiag, len_antidiag, N)
        )
        if apply_attn_mask:
            # num_antidiag, len_antidiag, 1
            mask = ~self.get_right_padded_antidiagonals(
                x.new_ones((1, freq, time), dtype=torch.bool)
            ).moveaxis(0, -1)
            mask = mask.float()
            # num_antidiag, len_antidiag, len_antidiag
            mask = torch.matmul(mask, mask.transpose(-1, -2)).bool()

            bs_head = batch * num_antidiag * self.tfs_nn.att_heads
            mask = (
                mask[None, :, None, ...]
                .expand(batch, -1, self.tfs_nn.att_heads, -1, -1)
                .reshape(bs_head, len_antidiag, len_antidiag)
            )
            x = self.tfs_nn(x, attn_mask=mask)
        else:
            x = self.tfs_nn(x)

        x = (
            x.reshape(batch, num_antidiag, len_antidiag, N)
            .moveaxis(-1, 1)
            .reshape(batch * N, num_antidiag, len_antidiag)
        )
        x = self.get_matrix_from_right_padded_antidiagonals(x, freq, time)
        return x.reshape(batch, N, freq, time)

    def get_right_padded_antidiagonals(self, input):
        bs, freq, time = input.size()

        short_side = min(time, freq)
        pad = short_side - 1
        input_pad_left = torch.nn.functional.pad(input[:, :short_side], (pad, 0))
        input_pad_below = torch.nn.functional.pad(
            input[..., -short_side:], (0, 0, 0, pad)
        )

        bs_stride_left = input_pad_left.size(1) * input_pad_left.size(2)
        bs_stride_below = input_pad_below.size(1) * input_pad_below.size(2)

        stride = time + pad - 1
        above = input_pad_left.as_strided(
            (bs, time - 1, short_side), (bs_stride_left, 1, stride), storage_offset=pad
        )
        below = input_pad_below.as_strided(
            (bs, freq, short_side), (bs_stride_below, pad + 1, pad), storage_offset=pad
        )
        antidiag_pad = torch.cat([above, below], dim=1)
        return antidiag_pad

    def get_matrix_from_right_padded_antidiagonals(self, antidiag_pad, dim1, dim2):
        assert antidiag_pad.size(1) == dim1 + dim2 - 1
        short_side = min(dim1, dim2)
        pad = short_side - 1

        bs_stride = antidiag_pad.size(1) * antidiag_pad.size(2)
        bs = antidiag_pad.size(0)
        if dim1 > dim2:
            above = torch.nn.functional.pad(antidiag_pad[:, :pad], (pad, 0))
            bs_stride_above = above.size(1) * above.size(2)
            stride = pad + short_side + 1
            above_left_padded = above.as_strided(
                (bs, pad, short_side), (bs_stride_above, stride, 1), storage_offset=0
            )
            recon = torch.cat([above_left_padded, antidiag_pad[:, pad:]], dim=1)
            recon = recon.as_strided(
                (bs, dim1, dim2), (bs_stride, dim2, dim2 - 1), storage_offset=dim2 - 1
            )
        else:
            below = torch.nn.functional.pad(antidiag_pad[:, -pad:], (pad, 0))
            bs_stride_below = below.size(1) * below.size(2)
            stride = pad + short_side - 1
            below_left_padded = below.as_strided(
                (bs, pad, short_side),
                (bs_stride_below, stride, 1),
                storage_offset=pad - 1,
            )
            recon = torch.cat([antidiag_pad[:, :-pad], below_left_padded], dim=1)
            recon = recon.as_strided(
                (bs, dim1, dim2), (bs_stride, dim1 + 1, dim1), storage_offset=0
            )
        return recon.contiguous()


class TFPSNet_Transformer(TFPSNet_Base):
    def __init__(
        self,
        input_size,
        bottleneck_size,
        output_size,
        tfps_blocks,
        # Transformer-related arguments
        rnn_type,
        hidden_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        bidirectional=True,
        norm_type="gLN",
    ):
        super().__init__(
            input_size,
            bottleneck_size,
            output_size,
            tfps_blocks,
            SingleTransformer,
            rnn_type,
            bottleneck_size,
            att_heads,
            hidden_size,
            dropout=dropout,
            activation=activation,
            bidirectional=bidirectional,
            norm=norm_type,
        )
        self.hidden_size = hidden_size

    def forward(self, input):
        B, N, F, T = input.shape
        output = self.layer_norm(input.reshape(B, N, -1))
        output = self.bottleneck_conv1x1(output).reshape(B, -1, F, T)  # B, BN, F, T
        for block in self.tfps_blocks:
            if isinstance(block, TFPSBlockType2):
                output = block(output, apply_attn_mask=True)
            else:
                output = block(output)

        output = self.output(output)  # B, output_size, F, T
        return output


class TFPSNet_Transformer_EDA(TFPSNet_Base):
    def __init__(
        self,
        input_size,
        bottleneck_size,
        output_size,
        tfps_blocks,
        # Transformer-related arguments
        rnn_type,
        hidden_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        bidirectional=True,
        norm_type="gLN",
        # EDA-related arguments
        i_eda_layer=4,
        num_eda_modules=1,
        # TSE-related arguments
        i_adapt_layer=4,
        adapt_layer_type="tfattn",
        adapt_enroll_dim=64,
        adapt_attention_dim=512,
        adapt_hidden_dim=512,
        adapt_softmax_temp=1,
    ):
        super().__init__(
            input_size,
            bottleneck_size,
            output_size,
            tfps_blocks,
            SingleTransformer,
            rnn_type,
            bottleneck_size,
            att_heads,
            hidden_size,
            dropout=dropout,
            activation=activation,
            bidirectional=bidirectional,
            norm=norm_type,
        )
        self.hidden_size = hidden_size

        # eda related params
        self.i_eda_layer = i_eda_layer
        if i_eda_layer is not None:
            self.sequence_aggregation = SequenceAggregation(bottleneck_size)
            self.eda = EncoderDecoderAttractor(bottleneck_size)

        # tse related params
        self.i_adapt_layer = i_adapt_layer
        if i_adapt_layer is not None:
            assert adapt_layer_type in ["tfattn", "tfattn_improved"]
            self.adapt_enroll_dim = adapt_enroll_dim
            self.adapt_layer_type = adapt_layer_type
            adapt_layer_params = {
                "attention_dim": adapt_attention_dim,
                "hidden_dim": adapt_hidden_dim,
                "softmax_temp": adapt_softmax_temp,
            }
            self.auxiliary_net = TFPSBlockType1(
                SingleTransformer,
                rnn_type,
                bottleneck_size,
                att_heads,
                hidden_size,
                dropout=dropout,
                activation=activation,
                bidirectional=bidirectional,
                norm=norm_type,
            )
            # prepare additional processing block
            if adapt_layer_type == "tfattn_improved":
                self.conditional_model = Conditional_TFPSNet_Transformer(
                    bottleneck_size,
                    adapt_hidden_dim,
                    bottleneck_size,
                    output_size,
                    (1, ),
                    SingleTransformer,
                    rnn_type,
                    bottleneck_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    bidirectional=bidirectional,
                    norm=norm_type,
                )
                adapt_layer_type = "tfattn"
            # load speaker selection module
            self.adapt_layer = make_adapt_layer(
                adapt_layer_type,
                indim=bottleneck_size,
                enrolldim=bottleneck_size,
                ninputs=1,
                adapt_layer_kwargs=adapt_layer_params,
            )
            self.layer_norm_enroll = ChannelwiseLayerNorm(adapt_enroll_dim)
            self.bottleneck_conv1x1_enroll = nn.Conv1d(adapt_enroll_dim, bottleneck_size, 1, bias=False)

    def forward(self, input, enroll_emb, num_spk=None):
        is_tse = enroll_emb is not None
        B, N, F, T = input.shape
        output = self.layer_norm(input.reshape(B, N, -1))
        output = self.bottleneck_conv1x1(output).reshape(B, -1, F, T)  # B, BN, F, T
        for i, block in enumerate(self.tfps_blocks):
            if isinstance(block, TFPSBlockType2):
                output = block(output, apply_attn_mask=True)
            else:
                output = block(output)
            # compute attractor
            if i == self.i_eda_layer:
                orig_B = B
                H = output.shape[-3]
                # aggregated_sequence = self.sequence_aggregation(output.permute(0, 2, 3, 1))
                aggregated_sequence = self.sequence_aggregation(output.transpose(-1, -3))
                attractors, probabilities = self.eda(aggregated_sequence, num_spk=num_spk)
                output = output[..., None, :, :, :] * attractors[..., :-1, :, None, None]  # [B, J, N, L, K]
                output = output.view(-1, H, F, T)
                B = output.shape[0]
            # target speaker extraction part
            if i == self.i_adapt_layer and is_tse:
                T_enroll = enroll_emb.shape[-1]
                enroll_emb = self.layer_norm(enroll_emb.reshape(orig_B, N, -1))
                enroll_emb = self.bottleneck_conv1x1(enroll_emb).reshape(orig_B, -1, F, T_enroll)  # B, BN, F, T
                # print("Grad", output.requires_grad, enroll_emb.requires_grad, flush=True)
                enroll_emb = self.auxiliary_net(enroll_emb)
                output = output.view(orig_B, -1, H, F, T)
                # print("Grad", output.requires_grad, enroll_emb.requires_grad, flush=True)
                # print(f"bef {output.shape}", flush=True)
                output = self.adapt_layer(output, enroll_emb)
                # print(f"af1 {output.shape}", flush=True)
                if self.adapt_layer_type == "tfattn_improved":
                    output = self.conditional_model(output, enroll_emb.mean(dim=(-1, -2), keepdim=True))
                    # print(f"af2 {output.shape}", flush=True)

        if self.i_eda_layer is None:
            probabilities = None
        output = self.output(output)  # B, output_size, dim1, dim2
        return output, probabilities


class Conditional_TFPSNet_Transformer(nn.Module):
    def __init__(
        self,
        enroll_size,
        conditioning_size,
        bottleneck_size,
        output_size,
        tfps_blocks,
        nn_module,
        *nn_args,
        **nn_kwargs,
    ):
        super().__init__()
        self.bottleneck_size = bottleneck_size
        self.output_size = output_size

        # dual-path transformer
        self.tfps_blocks = nn.ModuleList()
        self.film = nn.ModuleList()
        for block in tfps_blocks:
            if block == 1:
                self.tfps_blocks.append(
                    TFPSBlockType1(nn_module, *nn_args, **nn_kwargs)
                )
            elif block == 2:
                self.tfps_blocks.append(
                    TFPSBlockType2(nn_module, *nn_args, **nn_kwargs)
                )
            else:
                raise ValueError(f"TFPSBlock type ({block}) must be either 1 or 2")
            self.film.append(
                FiLM(
                    bottleneck_size,
                    enroll_size,
                    conditioning_size,
                )
            )
        self.hidden_size = bottleneck_size

    def forward(self, input, enroll_emb):
        B, BN, F, T = input.shape
        output = input
        for i, block in enumerate(self.tfps_blocks):
            output = self.film[i](output.transpose(-1, -3), enroll_emb.transpose(-1, -3)).transpose(-1, -3)
            if isinstance(block, TFPSBlockType2):
                output = block(output, apply_attn_mask=True)
            else:
                output = block(output)
        return output


class TFPSNet_RNN(TFPSNet_Base):
    def __init__(
        self,
        input_size,
        bottleneck_size,
        output_size,
        tfps_blocks,
        # RNN-related arguments
        rnn_type,
        hidden_size,
        dropout=0,
        bidirectional=True,
        norm_type="gLN",
    ):
        super().__init__(
            input_size,
            bottleneck_size,
            output_size,
            tfps_blocks,
            SingleRNN,
            rnn_type,
            bottleneck_size,
            hidden_size,
            dropout=dropout,
            bidirectional=bidirectional,
            norm=norm_type,
        )
        self.hidden_size = hidden_size

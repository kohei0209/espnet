# The implementation of DPTNet proposed in
# J. Chen, Q. Mao, and D. Liu, “Dual-path transformer network:
# Direct context-aware modeling for end-to-end monaural speech
# separation,” in Proc. ISCA Interspeech, 2020, pp. 2642–2646.
#
# Ported from https://github.com/ujscjj/DPTNet

import torch.nn as nn

from espnet2.enh.layers.tcn import choose_norm
from espnet.nets.pytorch_backend.nets_utils import get_activation
from espnet2.asr_transducer.encoder.modules.positional_encoding import RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet2.enh.layers.dprnn_eda import SequenceAggregation, EncoderDecoderAttractor


class TransformerLayer(nn.Module):
    """Container module of the (improved) Transformer proposed in [1].

    Reference:
        Dual-path transformer network: Direct context-aware modeling for end-to-end
        monaural speech separation; Chen et al, Interspeech 2020.

    Args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): Dimension of the input feature.
        att_heads (int): Number of attention heads.
        hidden_size (int): Dimension of the hidden state.
        dropout (float): Dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        bidirectional (bool, optional): True for bidirectional Inter-Chunk RNN
            (Intra-Chunk is always bidirectional).
        norm (str, optional): Type of normalization to use.
    """

    def __init__(
        self,
        input_size,
        att_heads,
        hidden_size,
        dropout=0.0,
        activation="relu",
        norm="gLN",
    ):
        super().__init__()

        self.att_heads = att_heads
        self.self_attn = nn.MultiheadAttention(input_size, att_heads, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm_attn = choose_norm(norm, input_size)

        activation = get_activation(activation)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, input_size),
        )
        self.norm_ff = choose_norm(norm, input_size)

    def forward(self, x, attn_mask=None):
        # (batch, seq, input_size)
        # ... -> (batch, input_size, seq) -> ...
        out = self.norm_attn(x.transpose(-1, -2)).transpose(-1, -2)
        # (batch, seq, input_size) -> (seq, batch, input_size)
        out = out.permute(1, 0, 2)
        # (seq, batch, input_size) -> (batch, seq, input_size)
        out = self.self_attn(out, out, out, attn_mask=attn_mask)[0].permute(1, 0, 2)
        out = self.dropout(out) + x
        # ... -> (batch, input_size, seq) -> ...
        out2 = self.norm_ff(out.transpose(-1, -2)).transpose(-1, -2)
        out2 = self.feed_forward(out2)
        out2 = self.dropout(out2) + out
        return out2


class Sepformer(nn.Module):
    """Dual-path transformer network.

    args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): dimension of the input feature.
            Input size must be a multiple of `att_heads`.
        hidden_size (int): dimension of the hidden state.
        output_size (int): dimension of the output size.
        att_heads (int): number of attention heads.
        dropout (float): dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        num_layers (int): number of stacked RNN layers. Default is 1.
        bidirectional (bool): whether the RNN layers are bidirectional. Default is True.
        norm_type (str): type of normalization to use after each inter- or
            intra-chunk Transformer block.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        num_intra_layers=4,
        num_inter_layers=1,
        norm_type="gLN",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # dual-path transformer
        self.intra_transformer = nn.ModuleList()
        self.inter_transformer = nn.ModuleList()
        for i in range(num_intra_layers):
            self.intra_transformer.append(
                TransformerLayer(
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                )
            )
        for i in range(num_inter_layers):
            self.inter_transformer.append(
                TransformerLayer(
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                )
            )
        # positional encoding
        self.positional_encoding = PositionalEncoding(input_size, dropout_rate=0)

        # output layer
        self.output = nn.Sequential(nn.PReLU(), nn.Linear(input_size, output_size))

    def forward(self, input):
        # input shape: batch, N, dim1, dim2
        # apply Transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        output = input
        # intra chunk processing
        # pos_enc!!!!!!!
        output = self.intra_chunk_process(output)
        output = self.inter_chunk_process(output)

        output = self.output(output.transpose(1, -1)).transpose(1, -1)  # B, output_size, dim1, dim2

        return output

    def intra_chunk_process(self, x):
        batch, N, chunk_size, n_chunks = x.size()
        out = x
        ### positional_encoding ###
        out = out.transpose(1, -1).reshape(batch * n_chunks, chunk_size, N)
        out = self.positional_encoding(out)
        for layer in self.intra_transformer:
            out = layer(out)
        out = out.reshape(batch, n_chunks, chunk_size, N).transpose(1, -1)
        return out + x

    def inter_chunk_process(self, x):
        batch, N, chunk_size, n_chunks = x.size()
        out = x
        ### positional_encoding ###
        out = out.permute(0, 2, 3, 1).reshape(batch * chunk_size, n_chunks, N)
        out = self.positional_encoding(out)
        for layer in self.inter_transformer:
            out = layer(out)
        out = out.reshape(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
        return out + x


class Sepformer_EDA(nn.Module):
    """Dual-path transformer network.

    args:
        rnn_type (str): select from 'RNN', 'LSTM' and 'GRU'.
        input_size (int): dimension of the input feature.
            Input size must be a multiple of `att_heads`.
        hidden_size (int): dimension of the hidden state.
        output_size (int): dimension of the output size.
        att_heads (int): number of attention heads.
        dropout (float): dropout ratio. Default is 0.
        activation (str): activation function applied at the output of RNN.
        num_layers (int): number of stacked RNN layers. Default is 1.
        bidirectional (bool): whether the RNN layers are bidirectional. Default is True.
        norm_type (str): type of normalization to use after each inter- or
            intra-chunk Transformer block.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        att_heads=4,
        dropout=0,
        activation="relu",
        num_intra_layers_before_eda=4,
        num_inter_layers_before_eda=2,
        num_intra_layers_after_eda=4,
        num_inter_layers_after_eda=2,
        num_channel_layers=2,
        norm_type="gLN",
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # dual-path block
        self.intra_transformer1 = nn.ModuleList()
        self.inter_transformer1 = nn.ModuleList()
        for i in range(num_intra_layers_before_eda):
            self.intra_transformer1.append(
                TransformerLayer(
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                )
            )
        for i in range(num_inter_layers_before_eda):
            self.inter_transformer1.append(
                TransformerLayer(
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                )
            )
        # triple-path block
        self.intra_transformer2 = nn.ModuleList()
        self.inter_transformer2 = nn.ModuleList()
        self.channel_transformer = nn.ModuleList()
        for i in range(num_intra_layers_after_eda):
            self.intra_transformer2.append(
                TransformerLayer(
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                )
            )
        for i in range(num_inter_layers_after_eda):
            self.inter_transformer2.append(
                TransformerLayer(
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                )
            )
        for i in range(num_channel_layers):
            self.channel_transformer.append(
                TransformerLayer(
                    input_size,
                    att_heads,
                    hidden_size,
                    dropout=dropout,
                    activation=activation,
                    norm=norm_type,
                )
            )
        # positional encoding
        self.positional_encoding = PositionalEncoding(input_size, dropout_rate=0)

        # output layer
        # assert input_size==output_size
        # self.output = nn.Sequential(nn.PReLU())
        self.output = nn.Sequential(nn.PReLU(), nn.Linear(input_size, output_size))
        # EDA-related modules
        self.sequence_aggregation = SequenceAggregation(input_size)
        self.eda = EncoderDecoderAttractor(input_size)

    def forward(self, input, num_spk):
        # input shape: batch, N, dim1, dim2
        # apply Transformer on dim1 first and then dim2
        # output shape: B, output_size, dim1, dim2
        # input = input.to(device)
        batch, hidden, dim1, dim2 = input.shape
        output = input
        # intra chunk processing
        output = self.intra_chunk_process(output, self.intra_transformer1)
        # inter chunk processing
        output = self.inter_chunk_process(output, self.intra_transformer1)
        '''
        # EDA-based speaker counting
        aggregated_sequence = self.sequence_aggregation(output.transpose(-1, -3))
        attractors, probabilities = self.eda(aggregated_sequence, num_spk=num_spk)
        output = output[..., None, :, :, :] * attractors[..., :-1, :, None, None] # [B, J, N, L, K]
        output = output.reshape(-1, hidden, dim1, dim2)
        # triple_path_processing
        output = self.intra_chunk_process(output, self.intra_transformer2)
        output = self.inter_chunk_process(output, self.intra_transformer2)
        output = output.reshape(batch, -1, hidden, dim1, dim2)
        output = self.inter_channel_process(output, self.channel_transformer)
        output = self.output(output)  # B, output_size, dim1, dim2
        '''
        output = output.reshape(batch, -1, dim1, dim2)
        output = self.output(output.transpose(1, -1)).transpose(1, -1)
        probabilities = None
        return output, probabilities

    def intra_chunk_process(self, x, transformer_blocks):
        batch, N, chunk_size, n_chunks = x.size()
        out = x
        ### positional_encoding ###
        out = out.transpose(1, -1).reshape(batch * n_chunks, chunk_size, N)
        out = self.positional_encoding(out)
        for layer in transformer_blocks:
            out = layer(out)
        out = out.reshape(batch, n_chunks, chunk_size, N).transpose(1, -1)
        return out + x

    def inter_chunk_process(self, x, transformer_blocks):
        batch, N, chunk_size, n_chunks = x.size()
        out = x
        ### positional_encoding ###
        out = out.permute(0, 2, 3, 1).reshape(batch * chunk_size, n_chunks, N)
        out = self.positional_encoding(out)
        for layer in transformer_blocks:
            out = layer(out)
        out = out.reshape(batch, chunk_size, n_chunks, N).permute(0, 3, 1, 2)
        return out + x

    def inter_channel_process(self, x, transformer_blocks):
        batch, num_spk, N, chunk_size, n_chunks = x.size()
        out = x
        ### not use the positional_encoding ###
        out = out.permute(0, 3, 4, 1, 2).reshape(batch * chunk_size * n_chunks, num_spk, N)
        for layer in transformer_blocks:
            out = layer(out)
        out = out.reshape(batch, chunk_size, n_chunks, num_spk, N).permute(0, 3, 4, 1, 2)
        return out + x
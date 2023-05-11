# noqa E501: Ported from https://github.com/BUTSpeechFIT/speakerbeam/blob/main/src/models/adapt_layers.py
# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

from functools import partial

import torch
import torch.nn as nn


def make_adapt_layer(type, indim, enrolldim, ninputs=1, adapt_layer_kwargs={}):
    adapt_class = adaptation_layer_types.get(type)
    return adapt_class(indim, enrolldim, ninputs, **adapt_layer_kwargs)


def into_tuple(x):
    """Transforms tensor/list/tuple into tuple."""
    if isinstance(x, list):
        return tuple(x)
    elif isinstance(x, torch.Tensor):
        return (x,)
    elif isinstance(x, tuple):
        return x
    else:
        raise ValueError("x should be tensor, list of tuple")


def into_orig_type(x, orig_type):
    """Inverts into_tuple function."""
    if orig_type is tuple:
        return x
    if orig_type is list:
        return list(x)
    if orig_type is torch.Tensor:
        return x[0]
    else:
        assert False


class ConcatAdaptLayer(nn.Module):
    def __init__(self, indim, enrolldim, ninputs=1):
        super().__init__()
        self.ninputs = ninputs
        self.transform = nn.ModuleList(
            [nn.Linear(indim + enrolldim, indim) for _ in range(ninputs)]
        )

    def forward(self, main, enroll):
        """ConcatAdaptLayer forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs

        out = []
        for transform, main0, enroll0 in zip(self.transform, main, enroll):
            out.append(
                transform(
                    torch.cat(
                        (main0, enroll0[:, :, None].expand(main0.shape)), dim=1
                    ).permute(0, 2, 1)
                ).permute(0, 2, 1)
            )
        return into_orig_type(tuple(out), orig_type)


class MulAddAdaptLayer(nn.Module):
    def __init__(self, indim, enrolldim, ninputs=1, do_addition=True):
        super().__init__()
        self.ninputs = ninputs
        self.do_addition = do_addition

        if do_addition:
            assert enrolldim == 2 * indim, (enrolldim, indim)
        else:
            assert enrolldim == indim, (enrolldim, indim)

    def forward(self, main, enroll):
        """MulAddAdaptLayer Forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        assert type(main) == type(enroll)
        orig_type = type(main)
        main, enroll = into_tuple(main), into_tuple(enroll)
        assert len(main) == len(enroll) == self.ninputs, (
            len(main),
            len(enroll),
            self.ninputs,
        )

        out = []
        for main0, enroll0 in zip(main, enroll):
            if self.do_addition:
                enroll0_mul, enroll0_add = torch.chunk(enroll0, 2, dim=1)
                out.append(enroll0_mul[:, :, None] * main0 + enroll0_add[:, :, None])
            else:
                out.append(enroll0[:, :, None] * main0)
        return into_orig_type(tuple(out), orig_type)


class AttentionAdaptLayer(nn.Module):
    def __init__(self, indim, enrolldim, ninputs=1, attention_dim=200, is_dualpath_process=False):
        super().__init__()
        self.ninputs = ninputs

        nonlinear = nn.PReLU()

        # attention modules
        self.mlp1 = nn.Sequential(
            nn.Linear(indim, attention_dim),
            nn.Linear(attention_dim, attention_dim),
            nonlinear,
            nn.Linear(attention_dim, attention_dim),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(indim, attention_dim),
            nn.Linear(attention_dim, attention_dim),
            nonlinear,
            nn.Linear(attention_dim, attention_dim),
        )

        self.mlp_aux = nn.Sequential(
            nn.Linear(indim, attention_dim),
            nn.Linear(attention_dim, attention_dim),
            nonlinear,
            nn.Linear(attention_dim, attention_dim),
        )

        self.W_v = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_iv = nn.Linear(attention_dim, attention_dim, bias=False)
        self.W_aux = nn.Linear(attention_dim, attention_dim, bias=False)
        self.w = nn.Linear(attention_dim, 1, bias=False)
        self.b = nn.Parameter(torch.randn(attention_dim))

        self.attention_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-2)
        self.alpha = 2
        self.indim = indim
        self.is_dualpath_process = is_dualpath_process

        assert enrolldim == indim, (enrolldim, indim)

    def forward(self, main, enroll=None):
        """MulAddAdaptLayer Forward.

        Args:
            main: tensor or tuple or list
                  activations in the main neural network, which are adapted
                  tuple/list may be useful when we want to apply the adaptation
                    to both normal and skip connection at once
            enroll: tensor or tuple or list
                    embedding extracted from enrollment
                    tuple/list may be useful when we want to apply the adaptation
                      to both normal and skip connection at once
        """
        out = []
        orig_type = type(main)
        is_tse = enroll is not None

        if not isinstance(main, tuple):
            main = (main, )
            enroll = (enroll, )

        for main0, enroll0 in zip(main, enroll):
            # main (not dual-path): (..., time, nspk, hidden)
            # main (dual-path): (..., chunk_size, num_chunk, nspk, hidden)
            # enroll(..., hidden)
            if is_tse:
                assert type(main0) == type(enroll0)
                if enroll0.ndim == 2:
                    enroll0 = enroll0[..., None, :]
                if self.is_dualpath_process:
                    batch, chunk_size, num_chunks, num_spk, hidden = main0.shape
                    mean_dim = (-3, -4)
                    enroll0 = enroll0[..., None, :]
                else:
                    batch, time, num_spk, hidden = main0.shape
                    mean_dim = (-3, )
                s_v = self.mlp1(main0) # (..., nspk, hidden)
                s_v2 = self.mlp2(main0).mean(dim=mean_dim, keepdim=True) # (..., nspk, hidden)
                s_aux = self.mlp_aux(enroll0) # (..., n_enroll, 1, hidden)

                # e: [batch, chunk_size, num_chunk, n_enroll, nspk, hidden]
                # a: [batch, chunk_size, num_chunk, n_enroll, nspk, 1]
                e1 = self.W_v(s_v) # (batch, chunk_size, num_chunk, nspk, hidden)
                e2 = self.W_iv(s_v2) # (..., nspk, hidden)
                e3 = self.W_aux(s_aux) # (batch, n_enroll, 1, hidden)
                e = e1[..., None, :, :] + e2[..., None, :, :] + e3[..., None, None, :, :, :] + self.b
                e = self.w(self.attention_activation(e))
                a = self.softmax(e*self.alpha)
                main0 = (a * main0[..., None, :, :]).sum(dim=-2)[..., 0, :]
            else:
                if self.is_dualpath_process:
                    batch, chunk_size, num_chunks, num_spk, hidden = main0.shape
                    zeros = main0.new_zeros((batch, chunk_size, num_chunks, 1, 1, 1))
                    ones = main0.new_ones((batch, chunk_size, num_chunks, 1, 1, 1))
                else:
                    batch, time, num_spk, hidden = main0.shape
                    zeros = main0.new_zeros((batch, time, 1, 1, 1))
                    ones = main0.new_ones((batch, time, 1, 1, 1))
                a = torch.cat((torch.cat((ones, zeros), dim=-2), torch.cat((zeros, ones), dim=-2)), dim=-3)
                main0 = (a * main0[..., None, :, :]).sum(dim=-2)
            out.append(main0)
        return into_orig_type(tuple(out), orig_type)

        '''
        #  return main
        out = []
        orig_type = type(main)
        is_tse = enroll is not None
        assert is_tse, "currently we assume TSE task"

        if not isinstance(main, tuple):
            main = (main, )
            enroll = (enroll, )

        for main0, enroll0 in zip(main, enroll):
            # main: (batch, nspk, hidden, time), enroll(batch, hidden)
            print(main0.shape, enroll0.shape)
            main0 = main0.transpose(-1, -2)
            main0 = main0[..., None, :, :] if main0.ndim == 3 else main0
            enroll0 = enroll0[..., None, :] if enroll0.ndim == 2 else enroll0
            if is_tse:
                # if behaving as TSE
                assert type(main0) == type(enroll0)
                s_v = self.mlp1(main0) # [B, I, T, F]
                s_v2 = self.mlp2(main0).mean(dim=-2, keepdim=True) # [B, I, 1, F]
                s_aux = self.mlp_aux(enroll0) # [B, S, 1, F]

                # e: [B, S, I, T, F], a: [B, S, I, T, 1]
                e1 = self.W_v(s_v)
                e2 = self.W_iv(s_v2)
                e3 = self.W_aux(s_aux)
                e = e1[..., None, :, :, :] + e2[..., None, :, :, :] + e3[..., None, None, :]
                e = self.w(self.attention_activation(e))
                a = self.softmax(e*self.alpha)
            # else:
            #     # force the attention weights to have 0 and 1 for each speaker
            #     B, I, T, F = Z.shape
            #     zeros, ones = Z.new_zeros((B, 1, 1, T, 1)), Z.new_ones((B, 1, 1, T, 1))
            #     a = torch.cat((torch.cat((ones, zeros), dim=2), torch.cat((zeros, ones), dim=2)), dim=1)
            # attention
            main0 = (a * main0[..., None, :, : ,:]).sum(dim=-3) # [B, S, T, F]
            B, S, T, F = main0.shape
            main0 = main0.reshape(-1, T, F).transpose(-1, -2)
            out.append(main0)
        return into_orig_type(tuple(out), orig_type)
        '''

# aliases for possible adaptation layer types
adaptation_layer_types = {
    "concat": ConcatAdaptLayer,
    "muladd": MulAddAdaptLayer,
    "mul": partial(MulAddAdaptLayer, do_addition=False),
    "attn": AttentionAdaptLayer,
}

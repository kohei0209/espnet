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

from espnet2.enh.layers.adapt_layers import make_adapt_layer


class ASENet(AbsExtractor, AbsSeparator):
    def __init__(
        self,
        input_dim: int,
        num_spk: int = 2,
        hidden_dim: int = 512,
        attention_dim: int = 200,
        dropout_p: float = 0.3,
        predict_noise: bool = False,
        # enrollment related arguments
        adapt_enroll_dim: int = 128,
        time_varying_adaptation: bool = True,
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
        self.lstm1 = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.linear1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.separator_linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.separator_linear2 = nn.Linear(hidden_dim, hidden_dim)

        # mask estimator modules
        self.lstm2 = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )
        self.linear2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.lstm3 = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )
        self.linear3 = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

        # tse related params
        self.adapt_enroll_dim = adapt_enroll_dim
        adapt_layer_kwargs={
            "is_dualpath_process": False,
            "attention_dim": attention_dim,
            "return_attn": True,
            "time_varying": time_varying_adaptation,
        }
        self.adapt_layer = make_adapt_layer(
            "attn",
            indim=hidden_dim,
            enrolldim=adapt_enroll_dim,
            ninputs=1,
            adapt_layer_kwargs=adapt_layer_kwargs,
        )


    def forward(
        self,
        input: Union[torch.Tensor, ComplexTensor],
        ilens: torch.Tensor,
        input_aux: torch.Tensor = None,
        ilens_aux: torch.Tensor = None,
        suffix_tag: str = "",
        num_spk: int = None,
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
        time_dim, freq_dim = feature.shape[1:]

        # separator part
        output, _ = self.lstm1(feature)
        output = self.linear1(output)
        output1 = self.separator_linear1(output)
        output2 = self.separator_linear2(output)
        output = torch.stack((output1, output2), dim=1) # batch, num_spk=2, time, freq

        batch, _, _, feature_dim = output.shape
        assert time_dim == output.shape[-2]
        if is_tse:
            enroll_emb = abs(input_aux)
            output, attn = self.adapt_layer(output, enroll_emb)
        output = output.reshape(-1, time_dim, feature_dim)

        # mask estimation part
        output, _ = self.lstm2(output)
        output = self.linear2(output)
        output, _ = self.lstm3(output)
        masks = self.linear3(output)
        masks = masks.reshape(batch, -1, time_dim, freq_dim).unbind(dim=-3)
        # assert len(masks) == num_spk

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
        if is_tse:
            others["attention"] = attn
        # if isinstance(input_aux, torch.Tensor):
        #     others["enroll_emb{}".format(suffix_tag)] = s_aux.detach()
        return masked, ilens, others

    @property
    def num_spk(self):
        return self._num_spk


def show_attention(attention):
    # attention: (n_spk, time)
    attention = attention.cpu().numpy()
    plt.show(attention)

if __name__ == "__main__":
    from pathlib import Path
    import matplotlib as plt
    model_dir = Path("./exp/enh_enhtse_asenet_2mix_raw")
    with open(model_dir / "config.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = ASENet(**config["extractor_conf"])
    model = model.load_state_dict(torch.load(model_dir / "valid.loss.best.path"))
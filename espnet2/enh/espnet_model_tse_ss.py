"""Enhancement model module."""
from typing import Dict, List, OrderedDict, Tuple

import torch
from typeguard import check_argument_types

from espnet2.enh.decoder.abs_decoder import AbsDecoder
from espnet2.enh.encoder.abs_encoder import AbsEncoder
from espnet2.enh.extractor.abs_extractor import AbsExtractor
from espnet2.enh.loss.criterions.tf_domain import FrequencyDomainLoss
from espnet2.enh.loss.criterions.time_domain import TimeDomainLoss
from espnet2.enh.loss.wrappers.abs_wrapper import AbsLossWrapper
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

EPS = torch.finfo(torch.get_default_dtype()).eps


def normalization(speech_mix, speech_ref=None, eps=1e-8):
    mean = speech_mix.mean(dim=-1, keepdim=True)
    std = speech_mix.std(dim=-1, keepdim=True)
    # standadization
    speech_mix = (speech_mix - mean) / (std + eps)
    if speech_ref is None:
        return speech_mix, mean, std
    else:
        speech_ref = [(ref - mean) / (std + eps) for ref in speech_ref]
        return speech_mix, speech_ref, mean, std

class ESPnetExtractionEnhancementModel(AbsESPnetModel):
    """Target Speaker Extraction Frontend model"""

    def __init__(
        self,
        encoder: AbsEncoder,
        extractor: AbsExtractor,
        decoder: AbsDecoder,
        loss_wrappers: List[AbsLossWrapper],
        num_spk: int = 1,
        share_encoder: bool = True,
        task: str = "enh_tse",
        normalization: bool = True,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.extractor = extractor
        self.decoder = decoder
        # Whether to share encoder for both mixture and enrollment
        self.share_encoder = share_encoder
        self.num_spk = num_spk
        self.normalization = normalization

        self.loss_wrappers = loss_wrappers
        names = [w.criterion.name for w in self.loss_wrappers]
        if len(set(names)) != len(names):
            raise ValueError("Duplicated loss names are not allowed: {}".format(names))
        for w in self.loss_wrappers:
            if getattr(w.criterion, "is_noise_loss", False):
                raise ValueError("is_noise_loss=True is not supported")
            elif getattr(w.criterion, "is_dereverb_loss", False):
                raise ValueError("is_dereverb_loss=True is not supported")

        # for multi-channel signal
        self.ref_channel = getattr(self.extractor, "ref_channel", -1)

        assert task in ["tse", "enh", "enh_tse"]
        self.task = task
        print(f"Task is {self.task}")
        print(f"Number of speakers {self.num_spk}")

    def forward(
        self,
        speech_mix: torch.Tensor,
        speech_mix_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech_mix: (Batch, samples) or (Batch, samples, channels)
            speech_ref1: (Batch, samples)
                        or (Batch, samples, channels)
            speech_ref2: (Batch, samples)
                        or (Batch, samples, channels)
            ...
            speech_mix_lengths: (Batch,), default None for chunk interator,
                            because the chunk-iterator does not have the
                            speech_lengths returned. see in
                            espnet2/iterators/chunk_iter_factory.py
            enroll_ref1: (Batch, samples_aux)
                                enrollment (raw audio or embedding) for speaker 1
            enroll_ref2: (Batch, samples_aux)
                                enrollment (raw audio or embedding) for speaker 2
            ...
            kwargs: "utt_id" is among the input.
        """
        # reference speech signal of each speaker
        assert "speech_ref1" in kwargs, "At least 1 reference signal input is required."
        speech_ref = [
            kwargs.get(
                f"speech_ref{spk + 1}",
                torch.zeros_like(kwargs["speech_ref1"]),
            )
            for spk in range(self.num_spk)
            if "speech_ref{}".format(spk + 1) in kwargs
        ]
        # remove dummy tensor with length of one
        speech_ref_len = len(speech_ref)
        speech_ref = [s for s in speech_ref if s.shape[-1]>1]
        for s in range(len(speech_ref), self.num_spk, 1):
            if "speech_ref{}".format(s + 1) in kwargs:
                kwargs.pop("speech_ref{}".format(s + 1))
        # (Batch, num_speaker, samples) or (Batch, num_speaker, samples, channels)
        speech_ref = torch.stack(speech_ref, dim=1)
        batch_size = speech_mix.shape[0]

        assert "enroll_ref1" in kwargs, "At least 1 enrollment signal is required."
        # enrollment signal for each speaker (as the target)
        enroll_ref = [
            # (Batch, samples_aux)
            kwargs["enroll_ref{}".format(spk + 1)]
            for spk in range(self.num_spk)
            if "enroll_ref{}".format(spk + 1) in kwargs
        ]
        # remove dummy tensor with length of one
        enroll_ref = [s for s in enroll_ref if s.shape[-1]>1]
        # remove keys from kwargs
        for s in range(len(enroll_ref), self.num_spk, 1):
            if "enroll_ref{}".format(s + 1) in kwargs:
                kwargs.pop("enroll_ref{}".format(s + 1))
        enroll_ref_lengths = [
            # (Batch,)
            kwargs.get(
                "enroll_ref{}_lengths".format(spk + 1),
                torch.ones(batch_size).int().fill_(enroll_ref[spk].size(1)),
            )
            for spk in range(self.num_spk)
            if "enroll_ref{}".format(spk + 1) in kwargs
        ]

        if "noise_ref1" in kwargs:
            # noise signal (optional, required when using beamforming-based
            # frontend models)
            noise_ref = [
                kwargs["noise_ref{}".format(n + 1)] for n in range(self.num_noise_type)
            ]
            # (Batch, num_noise_type, samples) or
            # (Batch, num_noise_type, samples, channels)
            noise_ref = torch.stack(noise_ref, dim=1)
        else:
            noise_ref = None

        # dereverberated (noisy) signal
        # (optional, only used for frontend models with WPE)
        if "dereverb_ref1" in kwargs:
            # noise signal (optional, required when using
            # frontend models with beamformering)
            dereverb_speech_ref = [
                kwargs["dereverb_ref{}".format(n + 1)]
                for n in range(self.num_spk)
                if "dereverb_ref{}".format(n + 1) in kwargs
            ]
            assert len(dereverb_speech_ref) in (1, self.num_spk), len(
                dereverb_speech_ref
            )
            # (Batch, N, samples) or (Batch, N, samples, channels)
            dereverb_speech_ref = torch.stack(dereverb_speech_ref, dim=1)
        else:
            dereverb_speech_ref = None

        batch_size = speech_mix.shape[0]

        speech_lengths = (
            speech_mix_lengths
            if speech_mix_lengths is not None
            else torch.ones(batch_size).int().fill_(speech_mix.shape[1])
        )
        assert speech_lengths.dim() == 1, speech_lengths.shape
        # Check that batch_size is unified
        assert speech_mix.shape[0] == speech_ref.shape[0] == speech_lengths.shape[0], (
            speech_mix.shape,
            speech_ref.shape,
            speech_lengths.shape,
        )
        for aux in enroll_ref:
            assert aux.shape[0] == speech_mix.shape[0], (aux.shape, speech_mix.shape)

        # for data-parallel
        speech_ref = speech_ref[..., : speech_lengths.max()].unbind(dim=1)
        if noise_ref is not None:
            noise_ref = noise_ref[..., : speech_lengths.max()].unbind(dim=1)
        if dereverb_speech_ref is not None:
            dereverb_speech_ref = dereverb_speech_ref[..., : speech_lengths.max()]
            dereverb_speech_ref = dereverb_speech_ref.unbind(dim=1)

        speech_mix = speech_mix[:, : speech_lengths.max()]
        enroll_ref = [
            enroll_ref[spk][:, : enroll_ref_lengths[spk].max()]
            for spk in range(len(enroll_ref))
        ]
        assert len(speech_ref) == len(enroll_ref), (len(speech_ref), len(enroll_ref))

        num_spk = len(speech_ref)
        import random
        if self.task == "tse":
            is_tse = True
        elif self.task == "enh":
            is_tse = False
        else: # randomly select TSE or Enhancement
            is_tse = random.random() > 0.5
        if is_tse:
            spk_idx = random.randint(0, len(speech_ref)-1)
            speech_ref = [speech_ref[spk_idx]]
            enroll_ref, enroll_ref_lengths = [enroll_ref[spk_idx]], [enroll_ref_lengths[spk_idx]]
        else:
            enroll_ref = None

        # model forward
        # num_spk = len(speech_ref) if self.training else None
        speech_pre, feature_mix, feature_pre, others = self.forward_enhance(
            speech_mix, speech_lengths, enroll_ref, enroll_ref_lengths, num_spk=num_spk, is_tse=is_tse,
            apply_normalization=self.normalization,
        )

        # loss computation
        loss, stats, weight, perm = self.forward_loss(
            speech_pre,
            speech_lengths,
            feature_mix,
            feature_pre,
            others,
            speech_ref,
            noise_ref,
            dereverb_speech_ref,
            num_spk=num_spk,
            is_tse=is_tse,
        )
        return loss, stats, weight

    def forward_enhance(
        self,
        speech_mix: torch.Tensor,
        speech_lengths: torch.Tensor,
        enroll_ref: torch.Tensor,
        enroll_ref_lengths: torch.Tensor,
        num_spk: int = None,
        is_tse: bool = True,
        apply_normalization: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if apply_normalization:
            speech_mix, mean, std = normalization(speech_mix)
            if is_tse:
                enroll_ref = [normalization(enroll_ref[spk])[0] for spk in range(len(enroll_ref))]

        feature_mix, flens = self.encoder(speech_mix, speech_lengths)

        # is_tse = enroll_ref is not None
        assert is_tse == (enroll_ref is not None), (is_tse, (enroll_ref is not None))
        if is_tse:
            if self.share_encoder:
                feature_aux, flens_aux = zip(
                    *[
                        self.encoder(enroll_ref[spk], enroll_ref_lengths[spk])
                        for spk in range(len(enroll_ref))
                    ]
                )
            else:
                feature_aux = enroll_ref
                flens_aux = enroll_ref_lengths

            feature_pre, _, others = zip(
                *[
                    self.extractor(
                        feature_mix,
                        flens,
                        feature_aux[spk],
                        flens_aux[spk],
                        suffix_tag=f"_spk{spk + 1}",
                        num_spk=num_spk,
                    )
                    for spk in range(len(enroll_ref))
                ]
            )
            others = {k: v for dic in others for k, v in dic.items()}
            if feature_pre[0] is not None:
                speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
            else:
                # some models (e.g. neural beamformer trained with mask loss)
                # do not predict time-domain signal in the training stage
                speech_pre = None
        else:
            # feature_pre, flens, others = self.extractor(feature_mix, flens, additional)
            feature_pre, flens, others = self.extractor(feature_mix, flens, num_spk=num_spk)
            if feature_pre is not None:
                # for models like SVoice that output multiple lists of separated signals
                pre_is_multi_list = isinstance(feature_pre[0], (list, tuple))
                if pre_is_multi_list:
                    speech_pre = [
                        [self.decoder(p, speech_lengths)[0] for p in ps]
                        for ps in feature_pre
                    ]
                else:
                    speech_pre = [self.decoder(ps, speech_lengths)[0] for ps in feature_pre]
            else:
                # some models (e.g. neural beamformer trained with mask loss)
                # do not predict time-domain signal in the training stage
                speech_pre = None
        if apply_normalization:
            speech_pre = [(pre + mean) * std for pre in speech_pre]
        return speech_pre, feature_mix, feature_pre, others

    def forward_loss(
        self,
        speech_pre: torch.Tensor,
        speech_lengths: torch.Tensor,
        feature_mix: torch.Tensor,
        feature_pre: torch.Tensor,
        others: OrderedDict,
        speech_ref: torch.Tensor,
        noise_ref: torch.Tensor = None,
        dereverb_speech_ref: torch.Tensor = None,
        num_spk: int = None,
        is_tse: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

        loss = 0.0
        stats = {}
        o = {}
        perm = None
        if is_tse:
            assert len(speech_pre) == len(speech_ref) == 1, "TSE should output only 1 source"
            for loss_wrapper in self.loss_wrappers:
                criterion = loss_wrapper.criterion
                if getattr(criterion, "only_for_test", False) and self.training:
                    continue

                if isinstance(criterion, TimeDomainLoss):
                    assert speech_pre is not None
                    sref, spre = self._align_ref_pre_channels(
                        speech_ref, speech_pre, ch_dim=2, force_1ch=True
                    )
                    # for the time domain criterions
                    l, s, o = loss_wrapper(sref, spre, {**others, **o})
                elif isinstance(criterion, FrequencyDomainLoss):
                    sref, spre = self._align_ref_pre_channels(
                        speech_ref, speech_pre, ch_dim=2, force_1ch=False
                    )
                    # for the time-frequency domain criterions
                    if criterion.compute_on_mask:
                        # compute loss on masks
                        tf_ref, tf_pre = self._get_speech_masks(
                            criterion,
                            feature_mix,
                            None,
                            speech_ref,
                            speech_pre,
                            speech_lengths,
                            others,
                        )
                    else:
                        # compute on spectrum
                        tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in sref]
                        tf_pre = [self.encoder(sp, speech_lengths)[0] for sp in spre]

                    l, s, o = loss_wrapper(tf_ref, tf_pre, {**others, **o})
                else:
                    raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

                loss += l * loss_wrapper.weight
                stats.update(s)

                if perm is None and "perm" in o:
                    perm = o["perm"]
            stats[f"tse_{num_spk}spk_loss"] = loss.detach()
            # for n in range(self.num_spk):
            #     if n == num_spk:
            #         stats[f"tse_{num_spk}spk_loss"] = loss.detach()
            #     else:
            #         stats[f"tse_{n}spk_loss"] = None
        else:
            # for calculating loss on estimated noise signals
            if getattr(self.extractor, "predict_noise", False):
                assert "noise1" in others, others.keys()
            if noise_ref is not None and "noise1" in others:
                for n in range(self.num_noise_type):
                    key = "noise{}".format(n + 1)
                    others[key] = self.decoder(others[key], speech_lengths)[0]
            # for calculating loss on dereverberated signals
            if getattr(self.extractor, "predict_dereverb", False):
                assert "dereverb1" in others, others.keys()
            if dereverb_speech_ref is not None and "dereverb1" in others:
                for spk in range(self.num_spk):
                    key = "dereverb{}".format(spk + 1)
                    if key in others:
                        others[key] = self.decoder(others[key], speech_lengths)[0]

            for loss_wrapper in self.loss_wrappers:
                criterion = loss_wrapper.criterion
                if getattr(criterion, "only_for_test", False) and self.training:
                    continue
                if getattr(criterion, "is_noise_loss", False):
                    if noise_ref is None:
                        raise ValueError(
                            "No noise reference for training!\n"
                            'Please specify "--use_noise_ref true" in run.sh'
                        )
                    signal_ref = noise_ref
                    signal_pre = [
                        others["noise{}".format(n + 1)] for n in range(self.num_noise_type)
                    ]
                elif getattr(criterion, "is_dereverb_loss", False):
                    if dereverb_speech_ref is None:
                        raise ValueError(
                            "No dereverberated reference for training!\n"
                            'Please specify "--use_dereverb_ref true" in run.sh'
                        )
                    signal_ref = dereverb_speech_ref
                    signal_pre = [
                        others["dereverb{}".format(n + 1)]
                        for n in range(self.num_noise_type)
                        if "dereverb{}".format(n + 1) in others
                    ]
                    if len(signal_pre) == 0:
                        signal_pre = None
                else:
                    signal_ref = speech_ref
                    signal_pre = speech_pre

                if isinstance(criterion, TimeDomainLoss):
                    assert signal_pre is not None
                    sref, spre = self._align_ref_pre_channels(
                        signal_ref, signal_pre, ch_dim=2, force_1ch=True
                    )
                    # for the time domain criterions
                    l, s, o = loss_wrapper(sref, spre, {**others, **o})
                elif isinstance(criterion, FrequencyDomainLoss):
                    sref, spre = self._align_ref_pre_channels(
                        signal_ref, signal_pre, ch_dim=2, force_1ch=False
                    )
                    # for the time-frequency domain criterions
                    if criterion.compute_on_mask:
                        # compute loss on masks
                        if getattr(criterion, "is_noise_loss", False):
                            tf_ref, tf_pre = self._get_noise_masks(
                                criterion,
                                feature_mix,
                                speech_ref,
                                signal_ref,
                                signal_pre,
                                speech_lengths,
                                others,
                            )
                        elif getattr(criterion, "is_dereverb_loss", False):
                            tf_ref, tf_pre = self._get_dereverb_masks(
                                criterion,
                                feature_mix,
                                noise_ref,
                                signal_ref,
                                signal_pre,
                                speech_lengths,
                                others,
                            )
                        else:
                            tf_ref, tf_pre = self._get_speech_masks(
                                criterion,
                                feature_mix,
                                noise_ref,
                                signal_ref,
                                signal_pre,
                                speech_lengths,
                                others,
                            )
                    else:
                        # compute on spectrum
                        tf_ref = [self.encoder(sr, speech_lengths)[0] for sr in sref]
                        # for models like SVoice that output multiple lists of
                        # separated signals
                        pre_is_multi_list = isinstance(spre[0], (list, tuple))
                        if pre_is_multi_list:
                            tf_pre = [
                                [self.encoder(sp, speech_lengths)[0] for sp in ps]
                                for ps in spre
                            ]
                        else:
                            tf_pre = [self.encoder(sp, speech_lengths)[0] for sp in spre]

                    l, s, o = loss_wrapper(tf_ref, tf_pre, {**others, **o})
                else:
                    raise NotImplementedError("Unsupported loss type: %s" % str(criterion))

                loss += l * loss_wrapper.weight
                stats.update(s)

                if perm is None and "perm" in o:
                    perm = o["perm"]

            # register loss value
            stats[f"enh_{num_spk}spk_loss"] = loss.clone().detach()
            # for n in range(self.num_spk):
            #     if n == num_spk:
            #         stats[f"enh_{num_spk}spk_loss"] = loss.detach()
            #     else:
            #         stats[f"enh_{n}spk_loss"] = None
            # compute EDA counting loss
            if "existance_probability" in others:
                import torch
                bce = torch.nn.BCELoss(reduction="none")
                exist, non_exist = others["existance_probability"][..., :num_spk], others["existance_probability"][..., num_spk]
                bce_loss_exist = bce(exist, torch.ones_like(exist)).sum(dim=-1)
                bce_loss_non_exist = bce(non_exist, torch.zeros_like(non_exist))
                # bce_loss_non_exist = bce(non_exist, torch.ones_like(non_exist)) # miss
                bce_loss = ((bce_loss_exist + bce_loss_non_exist) / (num_spk + 1)).mean()
                loss += 1 * bce_loss
                stats["attractor_loss"] = bce_loss.detach()
                stats["attractor_loss_exist"] = bce_loss_exist.mean().detach()
                stats["attractor_loss_nonexist"] = bce_loss_non_exist.mean().detach()
                stats["attractor_exist_prob"] = others["existance_probability"][..., :num_spk].mean().detach()
                stats["attractor_nonexist_prob"] = others["existance_probability"][..., num_spk].mean().detach()

        if self.training and isinstance(loss, float):
            raise AttributeError(
                "At least one criterion must satisfy: only_for_test=False"
            )
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        batch_size = speech_ref[0].shape[0]
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight, perm

    def _align_ref_pre_channels(self, ref, pre, ch_dim=2, force_1ch=False):
        if ref is None or pre is None:
            return ref, pre
        # NOTE: input must be a list of time-domain signals
        index = ref[0].new_tensor(self.ref_channel, dtype=torch.long)

        # for models like SVoice that output multiple lists of separated signals
        pre_is_multi_list = isinstance(pre[0], (list, tuple))
        pre_dim = pre[0][0].dim() if pre_is_multi_list else pre[0].dim()

        if ref[0].dim() > pre_dim:
            # multi-channel reference and single-channel output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
        elif ref[0].dim() < pre_dim:
            # single-channel reference and multi-channel output
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
        elif ref[0].dim() == pre_dim == 3 and force_1ch:
            # multi-channel reference and output
            ref = [r.index_select(ch_dim, index).squeeze(ch_dim) for r in ref]
            if pre_is_multi_list:
                pre = [
                    p.index_select(ch_dim, index).squeeze(ch_dim)
                    for plist in pre
                    for p in plist
                ]
            else:
                pre = [p.index_select(ch_dim, index).squeeze(ch_dim) for p in pre]
        return ref, pre

    def _get_speech_masks(
        self, criterion, feature_mix, noise_ref, speech_ref, speech_pre, ilens, others
    ):
        if noise_ref is not None:
            noise_spec = self.encoder(sum(noise_ref), ilens)[0]
        else:
            noise_spec = None
        masks_ref = criterion.create_mask_label(
            feature_mix,
            [self.encoder(sr, ilens)[0] for sr in speech_ref],
            noise_spec=noise_spec,
        )
        if "mask_spk1" in others:
            masks_pre = [
                others["mask_spk{}".format(spk + 1)] for spk in range(self.num_spk)
            ]
        else:
            masks_pre = criterion.create_mask_label(
                feature_mix,
                [self.encoder(sp, ilens)[0] for sp in speech_pre],
                noise_spec=noise_spec,
            )
        return masks_ref, masks_pre

    def collect_feats(
        self, speech_mix: torch.Tensor, speech_mix_lengths: torch.Tensor, **kwargs
    ) -> Dict[str, torch.Tensor]:
        # for data-parallel
        speech_mix = speech_mix[:, : speech_mix_lengths.max()]

        feats, feats_lengths = speech_mix, speech_mix_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

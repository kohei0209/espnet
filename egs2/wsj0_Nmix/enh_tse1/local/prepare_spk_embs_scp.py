import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
import copy
from pathlib import Path
import json

from espnet2.fileio.npy_scp import NpyScpWriter


def compute_fbank(
    wav_path, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0
):
    """Extract fbank.

    Simlilar to the one in wespeaker.dataset.processor,
    While integrating the wave reading and CMN.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    waveform = waveform * (1 << 15)
    mat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        dither=dither,
        sample_frequency=sample_rate,
        window_type="hamming",
        use_energy=False,
    )
    # CMN, without CVN
    mat = mat - torch.mean(mat, dim=0)
    return mat


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("scp", type=Path, help="scp file containing paths to utterances")
    parser.add_argument(
        "--onnx_path",
        type=str,
        required=True,
        help="Path to the pretrained model in ONNX format",
    )
    parser.add_argument(
        "--outdir", type=Path, required=True, help="Path to the output directory"
    )
    args = parser.parse_args()

    so = ort.SessionOptions()
    so.inter_op_num_threads = 1
    so.intra_op_num_threads = 1
    session = ort.InferenceSession(args.onnx_path, sess_options=so)

    writer = NpyScpWriter(args.outdir, f"{args.outdir}/embs.scp")
    # for valid and test
    if args.scp.suffix == ".scp":
        with open(args.scp, "r") as f:
            for i, line in enumerate(tqdm(f)):
                if i==2:
                    break
                if not line.strip():
                    continue
                uid, path = line.strip().split(maxsplit=1)
                if path != "dummy":
                    feats = compute_fbank(path)
                    feats = feats.unsqueeze(0).numpy()  # add batch dimension
                    embeddings = session.run(output_names=["embs"], input_feed={"feats": feats})
                    writer[uid] = np.squeeze(embeddings[0])
                else:
                    writer.fscp.write(f"{uid} dummy\n")
    # for training
    elif args.scp.suffix == ".json":
        with open(args.scp, "r") as f:
            data = json.load(f)
            new_data = copy.deepcopy(data)
        for k, (key, value) in enumerate(data.items()):
            for i, utt_data in enumerate(tqdm(value)):
                uid, path = utt_data
                feats = compute_fbank(path)
                feats = feats.unsqueeze(0).numpy()  # add batch dimension
                embeddings = session.run(output_names=["embs"], input_feed={"feats": feats})
                writer[uid] = np.squeeze(embeddings[0])
                new_data[key][i][-1] = writer.get_path(uid)
        with open(args.outdir / f"spk2emb.json", "w") as f:
            json.dump(new_data, f, indent=4)
    else:
        raise NotImplementedError()
    writer.close()
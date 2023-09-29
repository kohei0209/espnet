from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--libricss_root", type=Path)
parser.add_argument("--scp_output_dir", type=Path)
args = parser.parse_args()


libricss_wav_dir = args.libricss_root / "exp" / "data" / "monaural" / "utterances"
libricss_wav_dirs = libricss_wav_dir.iterdir()

# write scp file
scp = open(args.scp_output_dir / "wav.scp.bak", "w")
for wav_dir in libricss_wav_dirs:
    # avoid utterance_transcription.txt
    if not wav_dir.is_dir():
        continue
    for wav_path in wav_dir.iterdir():
        dirname = wav_dir.name
        filename = wav_path.stem
        key = f"{dirname}_{filename}"
        scp.write(f"{key} {str(wav_path)}\n")
scp.close()

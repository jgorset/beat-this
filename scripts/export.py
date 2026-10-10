"""Export the upstream final0 checkpoint and its exact preprocessing constants."""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch
from beat_this.inference import Audio2Frames


class BeatThisExport(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spect):
        result = self.model(spect)
        return result["beat"], result["downbeat"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(".tmp/export"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tracker = Audio2Frames(checkpoint_path="final0", device="cpu")
    model = BeatThisExport(tracker.model).eval()
    with torch.inference_mode():
        model(torch.zeros(1, 1500, 128))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            torch.onnx.export(model, torch.zeros(1, 1500, 128), args.out / "beat-this.onnx",
                              opset_version=17, dynamo=False, input_names=["spect"],
                              output_names=["beat", "downbeat"],
                              dynamic_axes={"spect": {1: "frames"}, "beat": {1: "frames"}, "downbeat": {1: "frames"}})
    onnx.checker.check_model(str(args.out / "beat-this.onnx"))
    filters = tracker.spect.spect_class.mel_scale.fb.numpy().astype(np.float32)
    window = tracker.spect.spect_class.spectrogram.window.numpy().astype(np.float32)
    assert filters.shape == (513, 128) and window.shape == (1024,)
    preprocessing = {"window": window.tolist(), "filters": [
        [[int(bin_), float(filters[bin_, band])] for bin_ in np.flatnonzero(filters[:, band])]
        for band in range(128)
    ]}
    (args.out / "preprocessing.json").write_text(json.dumps(preprocessing, separators=(",", ":")), encoding="utf8")
    model_bytes = (args.out / "beat-this.onnx").read_bytes()
    print(json.dumps({"checkpoint": "final0", "bytes": len(model_bytes), "sha256": hashlib.sha256(model_bytes).hexdigest()}))


if __name__ == "__main__":
    main()

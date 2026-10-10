# Third-party notices

This port adapts preprocessing, chunk aggregation and minimal postprocessing
from Beat This! by Francesco Foscarin, Jan Schlüter and Gerhard Widmer:
https://github.com/CPJKU/beat_this (package 1.1.0, checkpoint `final0`).
The ONNX export changes the model format, not its trained weights. The upstream
code and published weights are MIT-licensed; their copyright notice is retained
in LICENSE. Training recordings and annotations are not distributed here.

Reference paper: Foscarin, Schlüter and Widmer (2024), "Beat this! Accurate beat
tracking without DBN postprocessing", Proceedings of ISMIR 2024.

Runtime dependencies:

* ONNX Runtime Web 1.23.2, Microsoft Corporation, MIT:
  https://github.com/microsoft/onnxruntime
* fft.js 4.0.4, Fedor Indutny, MIT:
  https://github.com/indutny/fft.js

These dependencies are installed separately and retain their licenses in their
packages. The demo uses unmodified daisyUI CSS under MIT and Tailwind CSS under
MIT, served from jsDelivr. No dependency source is vendored into this package.

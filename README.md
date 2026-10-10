# Beat This!

Beat This! finds beats and downbeats in your browser. Give it some audio, get the
timestamps, and keep the audio on your machine. Pretty neat.

This is a TypeScript port of [Beat This!](https://github.com/CPJKU/beat_this) by
Francesco Foscarin, Jan Schlüter and Gerhard Widmer. It runs their `final0` model
with ONNX Runtime Web, including preprocessing and postprocessing.

## Installation

```sh
npm install https://github.com/jgorset/beat-this/releases/download/v1.0.0/jgorset-beat-this-1.0.0.tgz
```

Download [the model](https://github.com/jgorset/beat-this/releases/download/v1.0.0/beat-this.onnx)
and serve it from your application. It's about 83 MB, so you probably don't want
to load it until someone needs it. The package itself is small.

## Usage

```ts
import { createBeatThis } from '@jgorset/beat-this';

const tracker = await createBeatThis({
    modelUrl: '/models/beat-this.onnx',
    executionProvider: 'webgpu',
});

try {
    // Mono audio sampled at 22,050 Hz.
    const audio = new Float32Array(/* your audio samples */);
    const { beats, downbeats } = await tracker.analyze(audio);

    console.log(beats);     // Seconds from the start of the audio.
    console.log(downbeats); // The first beat of each bar.
} finally {
    await tracker.dispose();
}
```

Keep the tracker around if you're analyzing more than one recording. Loading the
model for every song would be a little rude to your computer.

Decoding files, mixing channels and resampling audio are up to your application.
The tracker accepts mono PCM at exactly 22,050 Hz. Run it in a Web Worker to keep
your interface responsive; the demo shows how.

### Long recordings

You don't have to keep the whole recording in memory. Supply a sample count and
a reader that returns exactly the requested samples:

```ts
const result = await tracker.analyze({
    sampleCount,
    read: async (first, count) => readAudioSamples(first, count),
}, {
    signal: abortController.signal,
    onProgress: (stage, completed, total) => console.log(stage, completed, total),
});
```

Reads are sequential and cover roughly 30 seconds plus overlap. `beats` and
`downbeats` contain seconds; `beatConfidence` and `downbeatConfidence` contain
probabilities at 20 ms intervals. Confidence arrays still grow with the recording.

Cancellation takes effect between windows and after an in-flight inference
finishes. To stop immediately, terminate the worker. Wait for analysis to settle
before disposing the tracker, and run one analysis at a time per tracker.

### WebGPU and WebAssembly

WebGPU is the default. Use HTTPS or localhost and a browser with WebGPU support.
You can select `executionProvider: 'wasm'` to run on the CPU instead. The package
uses the provider you select and reports errors if it can't run.

ONNX Runtime needs its matching runtime files. For Vite development, exclude
`onnxruntime-web` from `optimizeDeps`, as the demo does, so the runtime can find
its adjacent WASM file. If your application serves the files separately,
pass `wasmPaths: '/path/to/onnx-runtime/'` to `createBeatThis`. Serve the files from
`onnxruntime-web@1.23.2/dist`; runtime settings are shared by trackers in the same
JavaScript environment.

### Lower-level inference

Already decoding audio or running ONNX sessions? `extractFeatures` produces the
reference 128-band log-mel features, `inferBeatThis` runs overlapping windows on
an existing session, and `postprocess` converts beat/downbeat logits to seconds.
Their TypeScript declarations describe the inputs.

## Demo

```sh
npm ci
mkdir -p demo/public
curl -L https://github.com/jgorset/beat-this/releases/download/v1.0.0/beat-this.onnx -o demo/public/beat-this.onnx
npm run demo
```

Open the localhost address Vite prints, choose an audio file, and select GPU or
CPU inference. The demo uses
Web Audio to decode and resample it, then runs tracking in a worker. It keeps the
decoded file in memory, so use the streaming API for long recordings. Your audio
isn't uploaded.

## Model and accuracy

The release includes an ONNX export of the original `final0` weights, with
dynamic time dimensions, 1,500-frame windows, six-frame borders, and the upstream
`keep_first` aggregation and minimal postprocessor. Model weights are unchanged.

Different inference backends and resamplers can produce slightly different
timestamps near peak boundaries. That also happens between upstream CPU and GPU
inference. For reproducible comparisons, use identical PCM and the same backend.

The model checksum and upstream versions are in [model.json](model.json).
To export it yourself, use Python 3.10 or newer:

```sh
python -m pip install -r scripts/requirements.txt
python scripts/export.py --out .tmp/export
```

The export also includes `preprocessing.json` for inspection. Keep the package's
pinned constants when using the release model: generating the filterbank on
another platform can introduce small floating-point differences. If you change
the constants, update and validate the reference fixtures too.
Changing checkpoints may require changes to the port; this release targets `final0`.

## Contribute

* Fork the repository.
* Do your thing.
* Write a test.
* Open a pull request.

```sh
npm test
npm run demo:build
```

## License

MIT, including the original Beat This! code and published model weights. See
[LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the credits.

## I love you

Johannes Gorset made this port. You should [open an issue](https://github.com/jgorset/beat-this/issues)
if you can't get it to work. In fact, you should tell me what you're building anyway.

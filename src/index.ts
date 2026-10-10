import * as ort from 'onnxruntime-web/webgpu';
import { extractFeatures, HOP_SIZE, SAMPLE_RATE } from './dsp.js';
import { inferBeatThis, type Progress } from './inference.js';

export { extractFeatures, peakTimes, postprocess, HOP_SIZE, SAMPLE_RATE } from './dsp.js';
export { inferBeatThis, type Progress, type FeatureWindow } from './inference.js';

export type AudioSource = {
    /** Number of mono samples at 22,050 Hz. */
    sampleCount: number;
    read: (first: number, count: number) => Promise<Float32Array>;
};
export type BeatThisResult = {
    /** Event times in seconds. */
    beats: number[];
    downbeats: number[];
    /** Per-frame probabilities, starting at zero and spaced 20 ms apart. */
    beatConfidence: number[];
    downbeatConfidence: number[];
};
export type AnalyzeOptions = { onProgress?: Progress; signal?: AbortSignal };
export type BeatThis = {
    analyze: (audio: Float32Array | AudioSource, options?: AnalyzeOptions) => Promise<BeatThisResult>;
    dispose: () => Promise<void>;
};

/** Load a reusable tracker. Choose one execution provider; failures are surfaced. */
export async function createBeatThis(options: {
    modelUrl: string;
    executionProvider?: 'webgpu' | 'wasm';
    wasmPaths?: string;
}): Promise<BeatThis> {
    if (!options?.modelUrl) throw new TypeError('Provide a Beat This ONNX model URL.');
    if (options.executionProvider && !['webgpu', 'wasm'].includes(options.executionProvider)) {
        throw new TypeError('Choose the webgpu or wasm execution provider.');
    }
    if (options.wasmPaths !== undefined) ort.env.wasm.wasmPaths = options.wasmPaths;
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create(options.modelUrl, {
        executionProviders: [options.executionProvider ?? 'webgpu'], graphOptimizationLevel: 'all',
    });
    let busy = false, disposed = false;
    return {
        async analyze(audio, { onProgress, signal } = {}) {
            if (disposed) throw new Error('This tracker has been disposed.');
            if (busy) throw new Error('This tracker is already analyzing audio.');
            signal?.throwIfAborted();
            const source: AudioSource = audio instanceof Float32Array
                ? { sampleCount: audio.length, read: async (first, count) => audio.subarray(first, first + count) } : audio;
            if (!source || !Number.isSafeInteger(source.sampleCount) || source.sampleCount < 1 || typeof source.read !== 'function') {
                throw new TypeError('Provide nonempty mono audio at 22,050 Hz.');
            }
            busy = true;
            try {
                const frames = Math.floor(source.sampleCount / HOP_SIZE) + 1;
                const result = await inferBeatThis(session, frames, async (first, count) => {
                    signal?.throwIfAborted();
                    const begin = Math.max(0, first * HOP_SIZE - 1024);
                    const end = Math.min(source.sampleCount, (first + count) * HOP_SIZE + 1024);
                    const pcm = await source.read(begin, end - begin);
                    signal?.throwIfAborted();
                    if (!(pcm instanceof Float32Array) || pcm.length !== end - begin) throw new TypeError('Invalid audio window.');
                    return extractFeatures(pcm, first, count, source.sampleCount, begin);
                }, onProgress, signal);
                const beatConfidence: number[] = [], downbeatConfidence: number[] = [];
                for (let i = 0; i < frames; i++) {
                    beatConfidence[i] = 1 / (1 + Math.exp(-result.scores[2 * i]));
                    downbeatConfidence[i] = 1 / (1 + Math.exp(-result.scores[2 * i + 1]));
                }
                const duration = source.sampleCount / SAMPLE_RATE;
                return { beats: result.beats.filter(time => time < duration), downbeats: result.downbeats.filter(time => time < duration),
                    beatConfidence, downbeatConfidence };
            } finally { busy = false; }
        },
        async dispose() {
            if (busy) throw new Error('Wait for analysis to finish before disposing this tracker.');
            if (!disposed) { disposed = true; await session.release(); }
        },
    };
}

import * as ort from 'onnxruntime-web/webgpu';
import { BANDS, postprocess } from './dsp.js';

export type Progress = (stage: string, completed?: number, total?: number) => void;
export type FeatureWindow = (first: number, count: number) => Promise<Float32Array>;

/** Run reference-sized windows against a compatible Beat This! ONNX session. */
export async function inferBeatThis(session: ort.InferenceSession, frames: number,
    input: FeatureWindow, progress: Progress = () => {}, signal?: AbortSignal) {
    if (!Number.isSafeInteger(frames) || frames < 1) throw new RangeError('Invalid feature frame count.');
    signal?.throwIfAborted();
    const chunk = 1500, border = 6;
    const starts: number[] = [];
    for (let s = -border; s < frames - border; s += chunk - 2 * border) starts.push(s);
    if (frames > chunk - 2 * border) starts[starts.length - 1] = frames - (chunk - border);
    const beat = new Float32Array(frames), downbeat = new Float32Array(frames);
    beat.fill(-1000); downbeat.fill(-1000);
    // Keep the first prediction in the shifted final window's overlap, as upstream does.
    let completed = 0, writtenUntil = 0;
    for (const start of starts) {
        signal?.throwIfAborted();
        progress('Beat This inference', completed++, starts.length);
        const first = Math.max(0, start), stop = Math.min(start + chunk, frames);
        const left = Math.max(0, -start), right = Math.max(0, Math.min(border, start + chunk - frames));
        const count = stop - first + left + right;
        const features = await input(first, stop - first);
        signal?.throwIfAborted();
        if (!(features instanceof Float32Array) || features.length !== (stop - first) * BANDS
            || features.some(value => !Number.isFinite(value))) throw new TypeError('Invalid feature window.');
        const window = new Float32Array(count * BANDS);
        window.set(features, left * BANDS);
        const spect = new ort.Tensor('float32', window, [1, count, BANDS]);
        let result: ort.InferenceSession.ReturnType | undefined;
        try {
            result = await session.run({ spect });
            signal?.throwIfAborted();
            const b = result.beat?.data, d = result.downbeat?.data;
            if (!(b instanceof Float32Array) || !(d instanceof Float32Array) || b.length !== count || d.length !== count
                || b.some(value => !Number.isFinite(value)) || d.some(value => !Number.isFinite(value))) {
                throw new TypeError('Invalid Beat This model output.');
            }
            for (let i = Math.max(border, writtenUntil - start); i < count - border; i++) {
                beat[start + i] = b[i]; downbeat[start + i] = d[i];
            }
            writtenUntil = Math.max(writtenUntil, start + count - border);
        } finally {
            spect.dispose();
            for (const tensor of Object.values(result ?? {})) tensor.dispose();
        }
    }
    const scores = new Float32Array(frames * 2);
    for (let i = 0; i < frames; i++) { scores[2 * i] = beat[i]; scores[2 * i + 1] = downbeat[i]; }
    signal?.throwIfAborted();
    return { scores, ...postprocess(beat, downbeat) };
}

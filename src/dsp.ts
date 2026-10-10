import FFT from 'fft.js';
import constants from './preprocessing.json' with { type: 'json' };

export const SAMPLE_RATE = 22050;
export const HOP_SIZE = 441;
export const BANDS = 128;
const FFT_SIZE = 1024;
const window = Float32Array.from(constants.window);
const filters = constants.filters;

/** Extract the reference log-mel features from mono 22,050 Hz PCM. */
export function extractFeatures(pcm: Float32Array, first = 0,
    count = Math.floor(pcm.length / HOP_SIZE) + 1, totalSamples = pcm.length, pcmOffset = 0) {
    if (!Number.isSafeInteger(totalSamples) || totalSamples < 1 || !Number.isSafeInteger(first) || first < 0
        || !Number.isSafeInteger(count) || count < 1 || first + count > Math.floor(totalSamples / HOP_SIZE) + 1
        || !Number.isSafeInteger(pcmOffset) || pcmOffset < 0 || pcmOffset + pcm.length > totalSamples) {
        throw new RangeError('Invalid audio or feature window.');
    }
    for (const sample of pcm) if (!Number.isFinite(sample)) throw new TypeError('Audio samples must be finite.');
    const fft = new FFT(FFT_SIZE);
    const samples = new Float64Array(FFT_SIZE);
    const spectrum = new Float64Array(FFT_SIZE * 2);
    const magnitudes = new Float32Array(FFT_SIZE / 2 + 1);
    const output = new Float32Array(count * BANDS);
    for (let frame = 0; frame < count; frame++) {
        for (let i = 0; i < FFT_SIZE; i++) {
            let index = (first + frame) * HOP_SIZE - FFT_SIZE / 2 + i;
            if (index < 0) index = -index;
            if (index >= totalSamples) index = 2 * totalSamples - 2 - index;
            if (index >= 0 && index < totalSamples && (index < pcmOffset || index >= pcmOffset + pcm.length)) {
                throw new RangeError('Audio window is missing required samples.');
            }
            // Very short recordings retain the reference port's zero padding beyond reflection.
            samples[i] = (pcm[index - pcmOffset] ?? 0) * window[i];
        }
        fft.realTransform(spectrum, samples);
        fft.completeSpectrum(spectrum);
        for (let bin = 0; bin < magnitudes.length; bin++) {
            const real = spectrum[2 * bin], imaginary = spectrum[2 * bin + 1];
            magnitudes[bin] = Math.sqrt(real * real + imaginary * imaginary);
        }
        for (let band = 0; band < BANDS; band++) {
            let value = 0;
            for (const [bin, weight] of filters[band]) value += magnitudes[bin] * weight;
            output[frame * BANDS + band] = Math.log1p(1000 * value / Math.sqrt(FFT_SIZE));
        }
    }
    return output;
}

/** Positive seven-frame local maxima, with the upstream plateau deduplication. */
export function peakTimes(logits: Float32Array) {
    const peaks: number[] = [];
    for (let i = 0; i < logits.length; i++) {
        if (!Number.isFinite(logits[i])) throw new TypeError('Model scores must be finite.');
        if (logits[i] <= 0) continue;
        let maximum = true;
        for (let j = Math.max(0, i - 3); j <= Math.min(logits.length - 1, i + 3); j++) {
            if (logits[j] > logits[i]) maximum = false;
        }
        if (maximum) peaks.push(i);
    }
    const result: number[] = [];
    for (let i = 0; i < peaks.length;) {
        let j = i + 1, mean = peaks[i];
        while (j < peaks.length && peaks[j] - mean <= 1) {
            mean += (peaks[j] - mean) / (j - i + 1);
            j++;
        }
        result.push(peaks.slice(i, j).reduce((a, b) => a + b, 0) / (j - i) / 50);
        i = j;
    }
    return result;
}

/** Beat and downbeat times in seconds; downbeats snap to the nearest beat. */
export function postprocess(beat: Float32Array, downbeat: Float32Array) {
    if (beat.length !== downbeat.length) throw new RangeError('Beat and downbeat scores must have equal lengths.');
    const beats = peakTimes(beat), downbeats = peakTimes(downbeat);
    if (!beats.length) return { beats, downbeats };
    return { beats, downbeats: [...new Set(downbeats.map(time => {
        let best = beats[0];
        for (const candidate of beats) if (Math.abs(candidate - time) < Math.abs(best - time)) best = candidate;
        return best;
    }))] };
}

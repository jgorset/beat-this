import { BANDS } from './dsp.js';
import model from './changes.json' with { type: 'json' };

const FRAME_RATE = 50;
/** Seconds between the moments tested for a song change. */
const STEP = 0.5;
const COLUMNS = model.features;

/** scipy's one-dimensional filters with mode='nearest': the window for size n starts n // 2 before each element. */
function windowed(values: ArrayLike<number>, size: number, reduce: (window: Float64Array) => number) {
    const n = values.length, out = new Float64Array(n), window = new Float64Array(size), before = size >> 1;
    for (let i = 0; i < n; i++) {
        for (let k = 0; k < size; k++) window[k] = values[Math.min(Math.max(i - before + k, 0), n - 1)];
        out[i] = reduce(window);
    }
    return out;
}
export const maximumFilter = (values: ArrayLike<number>, size: number) => windowed(values, size, w => w.reduce((a, b) => Math.max(a, b)));
export const minimumFilter = (values: ArrayLike<number>, size: number) => windowed(values, size, w => w.reduce((a, b) => Math.min(a, b)));
export const uniformFilter = (values: ArrayLike<number>, size: number) => windowed(values, size, w => w.reduce((a, b) => a + b) / size);
export const medianFilter = (values: ArrayLike<number>, size: number) => windowed(values, size, w => w.slice().sort()[size >> 1]);

export function median(values: ArrayLike<number>) {
    const sorted = Float64Array.from(values).sort(), middle = sorted.length >> 1;
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Index of the first element not below `value` in a sorted list. */
export function lowerBound(sorted: ArrayLike<number>, value: number) {
    let low = 0, high = sorted.length;
    while (low < high) {
        const middle = (low + high) >> 1;
        if (sorted[middle] < value) low = middle + 1; else high = middle;
    }
    return low;
}

/** The probability of a song change for one row of measurements, from the gradient-boosted trees. */
export function changeProbability(row: ArrayLike<number>) {
    let sum = model.baseline;
    for (let tree = 0; tree < model.start.length - 1; tree++) {
        let node = model.start[tree];
        while (model.feature[node] >= 0) {      // a leaf has no feature, and its output where a split has its threshold
            const value = row[model.feature[node]];
            const left = Number.isNaN(value) ? model.missingLeft[node] : value <= model.value[node];
            node = left ? model.left[node] : model.right[node];
        }
        sum += model.value[node];
    }
    return 1 / (1 + Math.exp(-sum));
}

/**
 * For every half second, how different the audio just before is from the audio just after:
 * the sound, the loudness, and whether the beat keeps its tempo and stays in step.
 */
export function changeFeatures(mel: Float32Array, beats: ArrayLike<number>) {
    const frames = mel.length / BANDS, n = Math.floor(frames / 5), width = BANDS;
    // Band levels at 10 Hz, and running sums of them and their squares.
    const sum = new Float64Array((n + 1) * width), squares = new Float64Array((n + 1) * width), energy = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        for (let band = 0; band < width; band++) {
            let level = 0;
            for (let k = 0; k < 5; k++) level = Math.fround(level + mel[(i * 5 + k) * width + band]);
            level = Math.fround(level / 5);
            sum[(i + 1) * width + band] = sum[i * width + band] + level;
            squares[(i + 1) * width + band] = squares[i * width + band] + level * level;
            energy[i] += level / width;
        }
    }
    const clamp = (value: number, high = n) => Math.min(Math.max(value, 0), high);
    const count = Math.max(0, Math.ceil((n / 10 - 2 * STEP) / STEP));
    const times = Float64Array.from({ length: count }, (_, k) => STEP + k * STEP);
    const at = Array.from(times, time => Math.round(time * 10));
    const column = () => new Float64Array(count);

    /** Distance between the band statistics of the `span` before and after each moment, and the two band means. */
    const distance = (span: number, means?: { before: Float64Array; after: Float64Array }) => {
        const out = column();
        for (let k = 0; k < count; k++) {
            const a0 = clamp(at[k] - span), a1 = clamp(at[k]), b1 = clamp(at[k] + span);
            const na = Math.max(a1 - a0, 1), nb = Math.max(b1 - a1, 1);
            let total = 0;
            for (let band = 0; band < width; band++) {
                const ma = (sum[a1 * width + band] - sum[a0 * width + band]) / na, mb = (sum[b1 * width + band] - sum[a1 * width + band]) / nb;
                const sa = Math.sqrt(Math.max(0, (squares[a1 * width + band] - squares[a0 * width + band]) / na - ma * ma));
                const sb = Math.sqrt(Math.max(0, (squares[b1 * width + band] - squares[a1 * width + band]) / nb - mb * mb));
                total += Math.abs(ma - mb) / (sa + sb + 0.3);
                if (means) { means.before[k * width + band] = ma; means.after[k * width + band] = mb; }
            }
            out[k] = a1 - a0 < span >> 1 || b1 - a1 < span >> 1 ? 0 : total / width;
        }
        return out;
    };
    const means = { before: new Float64Array(count * width), after: new Float64Array(count * width) };
    const sound3 = distance(30, means), sound8 = distance(80), sound1 = distance(10);

    // An abrupt switch: the 0.6 s before against the 0.6 s after.
    const fine = column(), loudStep = column(), brightStep = column();
    for (let k = 0; k < count; k++) {
        const before = clamp(at[k] - 6), after = clamp(at[k] + 6);
        let total = 0, loud = 0, bright = 0;
        for (let band = 0; band < width; band++) {
            const a = (sum[at[k] * width + band] - sum[before * width + band]) / 6, b = (sum[after * width + band] - sum[at[k] * width + band]) / 6;
            total += Math.abs(a - b);
            const step = means.before[k * width + band] - means.after[k * width + band];
            loud += step;
            if (band >= 72) bright += step;
        }
        fine[k] = total / width; loudStep[k] = Math.abs(loud / width); brightStep[k] = Math.abs(bright / (width - 72));
    }

    // Loudness: a dip or a silence around the moment.
    const lowest = minimumFilter(energy, 21), usual = medianFilter(energy, 201), floor = 0.25 * (n ? median(energy) : 0);
    const dip = column(), quiet = column(), quietLength = column(), run = new Float64Array(n);
    for (let i = 0; i < n;) {
        if (energy[i] >= floor) { i++; continue; }
        let end = i;
        while (end < n && energy[end] < floor) end++;
        run.fill((end - i) / 10, Math.max(0, i - 5), Math.min(n, end + 5));
        i = end;
    }
    for (let k = 0; k < count; k++) {
        const i = clamp(at[k], n - 1);
        dip[k] = usual[i] - lowest[i]; quiet[k] = lowest[i] < floor ? 1 : 0; quietLength[k] = run[i];
    }

    // The beat: the last 16 beats before against the first 16 after, within 30 s.
    const tempo = column(), tempoRaw = column(), phase = column(), beatsBefore = column(), beatsAfter = column(), gap = column();
    const gaps = (list: number[]) => list.slice(1).map((time, i) => time - list[i]);
    for (let k = 0; k < count; k++) {
        const index = lowerBound(beats, times[k]);
        const before = Array.prototype.slice.call(beats, Math.max(0, index - 16), index).filter((beat: number) => beat > times[k] - 30);
        const after = Array.prototype.slice.call(beats, index, index + 16).filter((beat: number) => beat < times[k] + 30);
        beatsBefore[k] = before.length; beatsAfter[k] = after.length;
        if (before.length < 5 || after.length < 5) { tempo[k] = tempoRaw[k] = phase[k] = gap[k] = NaN; continue; }
        const lengthBefore = median(gaps(before)), lengthAfter = median(gaps(after)), last = before[before.length - 1];
        const ratio = Math.abs(Math.log2(lengthBefore / lengthAfter));
        tempoRaw[k] = ratio;
        tempo[k] = Math.min(ratio, Math.abs(ratio - 1), Math.abs(ratio - Math.log2(1.5)));
        gap[k] = Math.min(40, (after[0] - last) / lengthBefore);
        const offsets = after.slice(0, 4).map((beat: number) => { const part = ((beat - last) / lengthBefore) % 1; return Math.min(part, 1 - part); });
        phase[k] = offsets.reduce((a: number, b: number) => a + b, 0) / offsets.length;
    }

    // Against the surrounding minute.
    const relative = (values: Float64Array) => { const smooth = uniformFilter(values, 120); return values.map((value, k) => value / (smooth[k] + 0.05)); };
    const columns: Record<string, Float64Array> = {
        snd3: sound3, snd8: sound8, snd1: sound1, fine, dip, quiet, tempo, tempo_raw: tempoRaw, phase, nbeat_a: beatsBefore, nbeat_b: beatsAfter,
        loud_step: loudStep, bright_step: brightStep, snd3_rel: relative(sound3), snd8_rel: relative(sound8), fine_rel: relative(fine),
        gap_beats: gap, quiet_len: quietLength,
    };
    const rows = Array.from({ length: count }, (_, k) => Float64Array.from(COLUMNS, name => columns[name][k]));
    return { times, rows };
}

/** scipy.signal.find_peaks with a minimum height and distance: the highest peaks win. */
export function findPeaks(values: ArrayLike<number>, height: number, distance: number) {
    const peaks: number[] = [];
    for (let i = 1; i < values.length - 1;) {
        if (values[i - 1] < values[i]) {
            let ahead = i + 1;
            while (ahead < values.length - 1 && values[ahead] === values[i]) ahead++;
            if (values[ahead] < values[i]) { peaks.push((i + ahead - 1) >> 1); i = ahead; continue; }
        }
        i++;
    }
    const tall = peaks.filter(peak => values[peak] >= height), keep = tall.map(() => true);
    const order = tall.map((_, i) => i).sort((a, b) => values[tall[a]] - values[tall[b]] || a - b);
    for (let rank = order.length - 1; rank >= 0; rank--) {
        const j = order[rank];
        if (!keep[j]) continue;
        for (let k = j - 1; k >= 0 && tall[j] - tall[k] < distance; k--) keep[k] = false;
        for (let k = j + 1; k < tall.length && tall[k] - tall[j] < distance; k++) keep[k] = false;
    }
    return tall.filter((_, i) => keep[i]);
}

/** Move a detected change to the exact moment the sound switches, or to the end of a silence. */
function refine(mel: Float32Array, time: number, quietBelow: number, search = 2.5) {
    const frames = mel.length / BANDS, centre = Math.round(time * FRAME_RATE), span = 25;
    const first = Math.max(30, centre - search * FRAME_RATE), last = Math.min(frames - 30, centre + search * FRAME_RATE);
    if (last <= first) return time;
    const level = (frame: number) => { let total = 0; for (let band = 0; band < BANDS; band++) total += mel[frame * BANDS + band]; return total / BANDS; };
    let lastQuiet = -1, quiet = 0;
    for (let frame = first; frame < last; frame++) if (level(frame) < quietBelow) { quiet++; lastQuiet = frame; }
    if (quiet >= 10) return (lastQuiet + 1) / FRAME_RATE;       // a silence: the new song starts where the sound returns
    let best = first, strongest = -1;
    const before = new Float64Array(BANDS), after = new Float64Array(BANDS);
    for (let frame = first; frame < last; frame++) {
        before.fill(0); after.fill(0);
        for (let k = 1; k <= span; k++) for (let band = 0; band < BANDS; band++) {
            before[band] += mel[(frame - k) * BANDS + band]; after[band] += mel[(frame + k - 1) * BANDS + band];
        }
        let novelty = 0;
        for (let band = 0; band < BANDS; band++) novelty += Math.abs(before[band] - after[band]);
        if (novelty > strongest) { strongest = novelty; best = frame; }
    }
    return best / FRAME_RATE;
}

/**
 * Song-change times in seconds.
 *
 * A change is only reported where the beat itself breaks (the tempo changes, or the new beat is out of
 * step with the old one) or there is a silence. A change in sound alone is usually a new section of the
 * same song, and cutting there would take away the context a quiet section needs to be tracked at all.
 */
export function detectChanges(mel: Float32Array, beats: ArrayLike<number>, options: { threshold?: number; minSong?: number } = {}) {
    const { threshold = 0.7, minSong = 30 } = options, frames = mel.length / BANDS;
    const { times, rows } = changeFeatures(mel, beats);
    if (!rows.length) return [];
    const column = (name: string) => COLUMNS.indexOf(name);
    const [tempo, phase, quiet, quietLength] = ['tempo', 'phase', 'quiet', 'quiet_len'].map(column);
    const smooth = uniformFilter(rows.map(changeProbability), 3);
    const broken = rows.map(row => row[tempo] > 0.03 || row[phase] > 0.15 || row[quiet] > 0 || row[quietLength] >= 0.3 ? 1 : 0);
    const near = maximumFilter(broken, 2 * Math.floor(3 / STEP) + 1);        // within 3 s
    const peaks = findPeaks(smooth.map((value, k) => near[k] > 0 ? value : 0), threshold, Math.max(1, Math.floor(minSong / STEP)));
    const levels = new Float32Array(frames);
    for (let frame = 0; frame < frames; frame++) {
        let total = 0;
        for (let band = 0; band < BANDS; band++) total += mel[frame * BANDS + band];
        levels[frame] = total / BANDS;
    }
    const quietBelow = 0.25 * median(levels), duration = frames / FRAME_RATE;
    return peaks.map(peak => refine(mel, times[peak], quietBelow)).filter(time => time > 5 && time < duration - 5);
}

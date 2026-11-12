import * as ort from 'onnxruntime-web/webgpu';
import { detectChanges, lowerBound, maximumFilter, median, medianFilter, uniformFilter } from './changes.js';
import { BANDS, peakTimes, postprocess } from './dsp.js';
import { inferBeatThis, type Progress } from './inference.js';

const FRAME_RATE = 50;
/**
 * Frames per model pass, and the distance between passes: 30 s, every 15 s. Closer passes cost more and
 * track no better; passes further apart lose the beat in quiet stretches, which need the chance of a pass
 * that happens to hear them.
 */
const SIZE = 1500, HOP = 750;

export type Scores = { beat: Float32Array; downbeat: Float32Array; permissive: Float32Array };
export type SongOptions = {
    /** Shortest song, in seconds. Changes closer together than this are merged. The default is 30. */
    minSong?: number;
    /** How sure the song-change model must be, from 0 to 1. The default is 0.7. */
    threshold?: number;
};

const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.min(Math.max(value, -30), 30)));
/** Round half to even, as numpy does. */
const round = (value: number) => { const down = Math.floor(value), rest = value - down; return rest > 0.5 || (rest === 0.5 && down % 2) ? down + 1 : down; };
const gaps = (times: ArrayLike<number>) => Float64Array.from({ length: Math.max(0, times.length - 1) }, (_, i) => times[i + 1] - times[i]);

/**
 * Run the model over frames first..last with passes that overlap, so every frame is heard with plenty of
 * context on both sides. The beat and downbeat scores of a frame come from the pass in which it is most
 * central. The permissive beat scores come from the pass that hears a beat most clearly around it; they
 * are only used to fill holes.
 */
export async function overlappedScores(session: ort.InferenceSession, mel: Float32Array, first = 0, last = mel.length / BANDS,
    progress: (passes: number) => void = () => {}, signal?: AbortSignal): Promise<Scores> {
    const frames = last - first;
    if (frames <= SIZE) {
        const result = await inferBeatThis(session, frames,
            async (start, count) => mel.slice((first + start) * BANDS, (first + start + count) * BANDS), undefined, signal);
        const beat = result.scores.filter((_, i) => i % 2 === 0), downbeat = result.scores.filter((_, i) => i % 2 === 1);
        progress(1);
        return { beat, downbeat, permissive: beat.slice() };
    }
    const starts: number[] = [];
    for (let start = 0; start < frames - SIZE; start += HOP) starts.push(start);
    starts.push(frames - SIZE);
    const beat = new Float32Array(frames).fill(-1000), downbeat = new Float32Array(frames).fill(-1000), permissive = new Float32Array(frames).fill(-1000);
    const central = new Float32Array(frames).fill(-1e9), clear = new Float32Array(frames).fill(-1e9);
    for (const start of starts) {
        signal?.throwIfAborted();
        const spect = new ort.Tensor('float32', mel.slice((first + start) * BANDS, (first + start + SIZE) * BANDS), [1, SIZE, BANDS]);
        let result: ort.InferenceSession.ReturnType | undefined;
        try {
            result = await session.run({ spect });
            signal?.throwIfAborted();
            const b = result.beat?.data, d = result.downbeat?.data;
            if (!(b instanceof Float32Array) || !(d instanceof Float32Array) || b.length !== SIZE || d.length !== SIZE
                || b.some(value => !Number.isFinite(value)) || d.some(value => !Number.isFinite(value))) {
                throw new TypeError('Invalid Beat This model output.');
            }
            // How clearly this pass hears a beat around each frame, smoothed over about 3 s.
            const clarity = uniformFilter(maximumFilter(Float32Array.from(b, sigmoid), 25), 150);
            for (let i = 0; i < SIZE; i++) {
                // Distance from the nearer edge of the pass; the true ends of the audio count as fully central.
                let centre = Math.fround(Math.min(i, SIZE - 1 - i) / (SIZE / 2));
                if ((start === 0 && i < SIZE / 2) || (start === frames - SIZE && i >= SIZE / 2)) centre = 1;
                if (centre > central[start + i]) { beat[start + i] = b[i]; downbeat[start + i] = d[i]; central[start + i] = centre; }
                let score = Math.fround(clarity[i] + 0.15 * centre);
                if ((start > 0 && i < 25) || (start < frames - SIZE && i >= SIZE - 25)) score = -1e8;      // never the very edge of a pass
                if (score > clear[start + i]) { permissive[start + i] = b[i]; clear[start + i] = score; }
            }
        } finally {
            spect.dispose();
            for (const tensor of Object.values(result ?? {})) tensor.dispose();
        }
        progress(1);
    }
    return { beat, downbeat, permissive };
}

/** How many model passes `overlappedScores` makes for this many frames. */
export const passCount = (frames: number) => frames <= SIZE ? 1 : Math.ceil((frames - SIZE) / HOP) + 1;

/**
 * Where the beats leave a hole (a gap much longer than the beats around it), fill it with the beats of
 * the permissive scores. The usual case is a quiet intro that only some passes hear a beat in.
 */
export function fillHoles(beats: number[], permissive: Float32Array, factor = 1.7) {
    const extra = peakTimes(permissive);
    if (!extra.length) return beats;
    if (beats.length < 4) return extra.length > beats.length + 8 ? extra : beats;
    const lengths = gaps(beats), usual = medianFilter(lengths.map(length => Math.min(length, 1.5)), 9);
    const step = extra.length > 2 ? median(gaps(extra)) : 0.5;
    const added: number[] = [];
    for (let i = 0; i <= beats.length; i++) {
        const from = i > 0 ? beats[i - 1] : -1e9, to = i < beats.length ? beats[i] : 1e9;
        const width = i > 0 && i < beats.length ? lengths[i - 1] : 1e9;
        if (width > factor * usual[Math.min(Math.max(i - 1, 0), usual.length - 1)]) {
            for (const time of extra) if (time > from + 0.6 * step && time < to - 0.6 * step) added.push(time);
        }
    }
    return added.length ? [...beats, ...added].sort((a, b) => a - b) : beats;
}

/**
 * Choose which beats start a bar.
 *
 * Bars are counted steadily (1-2-3-4, or 1-2-3) and the count only slips where the evidence is strong:
 * a slip costs `slip`. Where a beat is missing (a gap of two or three beat lengths) the count advances by
 * that many, so one dropped beat does not throw the bars off. At every time in `resets` (the song
 * changes) the count may restart for free, so a wrong count is not carried into the next song.
 */
export function countBars(beats: number[], downbeat: Float32Array, resets: number[] = [], slip = 16) {
    const count = beats.length;
    if (!count) return [];
    const states = [3, 4].flatMap(meter => Array.from({ length: meter }, (_, position) => ({ meter, position })));
    const prior = states.map(state => state.meter === 3 ? -3 : 0);       // four in a bar unless the evidence says three
    const transition = (advance: number) => states.map((from, a) => states.map((to, b) =>
        to.meter === from.meter ? (to.position === (from.position + advance) % from.meter ? 0 : -slip) : -slip + prior[b] - prior[a]));
    const steady = [1, 2, 3, 4].map(transition), free = states.map(() => prior);
    const lengths = gaps(beats), usual = count > 4 ? medianFilter(lengths.map(length => Math.min(length, 1.5)), 9) : lengths;
    const reset = new Uint8Array(count);
    for (const time of resets) { const index = lowerBound(beats, time); if (index > 0 && index < count) reset[index] = 1; }
    const emission = (beat: number) => {
        const frame = Math.min(Math.max(round(beats[beat] * FRAME_RATE), 0), downbeat.length - 1);
        let chance = 0;
        for (let k = -3; k <= 3; k++) chance = Math.max(chance, sigmoid(downbeat[Math.min(Math.max(frame + k, 0), downbeat.length - 1)]));
        chance = Math.min(Math.max(chance, 1e-4), 1 - 1e-4);
        return states.map(state => Math.log(state.position === 0 ? chance : 1 - chance));
    };
    let score = emission(0).map((value, state) => value + prior[state]);
    const back = new Int8Array(count * states.length);
    for (let beat = 1; beat < count; beat++) {
        const advance = count > 4 ? Math.min(Math.max(round(lengths[beat - 1] / usual[beat - 1]), 1), 4) : 1;
        const moves = reset[beat] ? free : steady[advance - 1], heard = emission(beat);
        score = states.map((_, to) => {
            let best = 0;
            for (let from = 1; from < states.length; from++) if (score[from] + moves[from][to] > score[best] + moves[best][to]) best = from;
            back[beat * states.length + to] = best;
            return score[best] + moves[best][to] + heard[to];
        });
    }
    let state = score.indexOf(Math.max(...score));
    const bars: number[] = [];
    for (let beat = count - 1; beat >= 0; beat--) {
        if (states[state].position === 0) bars.push(beats[beat]);
        state = back[beat * states.length + state];
    }
    return bars.reverse();
}

/** How much beat the model hears in a stretch: the summed probability of its beats. */
function confidence(scores: Float32Array) {
    if (scores.length < 10) return 0;
    let total = 0;
    for (const time of peakTimes(scores)) total += sigmoid(scores[Math.min(Math.max(round(time * FRAME_RATE), 0), scores.length - 1)]);
    return total;
}

/**
 * Per song, use the scores made from that song alone unless the scores made with its neighbours hear
 * clearly more beat. A quiet intro can be impossible to track on its own yet easy with the next section
 * as context, and cutting it off would lose its beats.
 */
export function preferContext(whole: Scores, alone: Scores, changes: number[], ratio = 1.3): Scores {
    const frames = whole.beat.length, cuts = [0, ...changes.map(time => round(time * FRAME_RATE)), frames];
    const out = { beat: alone.beat.slice(), downbeat: alone.downbeat.slice(), permissive: alone.permissive.slice() };
    for (let i = 0; i < cuts.length - 1; i++) {
        const [from, to] = [cuts[i], cuts[i + 1]];
        if (confidence(whole.permissive.subarray(from, to)) > ratio * confidence(alone.permissive.subarray(from, to)) + 4) {
            for (const key of ['beat', 'downbeat', 'permissive'] as const) out[key].set(whole[key].subarray(from, to), from);
        }
    }
    return out;
}

/** Beats and bars from the final scores: pick the beats, fill the holes, count the bars per song. */
export function decodeSongs(scores: Scores, changes: number[]) {
    const beats = fillHoles(postprocess(scores.beat, scores.downbeat).beats, scores.permissive);
    return { beats, downbeats: countBars(beats, scores.downbeat, changes) };
}

/**
 * Track beats and bars in audio that holds several songs, from its log-mel features.
 *
 * 1. Run the model over everything with overlapping passes and find where the songs change.
 * 2. Run it again on every song by itself, so no pass hears two songs at once.
 * 3. Pick the beats, fill the holes, and count the bars per song, starting again at every change.
 */
export async function inferSongs(session: ort.InferenceSession, mel: Float32Array, progress: Progress = () => {},
    signal?: AbortSignal, options: SongOptions = {}) {
    const frames = mel.length / BANDS;
    if (!Number.isSafeInteger(frames) || frames < 1) throw new RangeError('Invalid feature frame count.');
    // The second round is at most a little longer than the first, so plan for twice the first.
    let completed = 0;
    const total = 2 * passCount(frames), step = (passes: number) => progress('Beat This inference', completed += passes, total);
    progress('Beat This inference', 0, total);
    let scores = await overlappedScores(session, mel, 0, frames, step, signal);
    const first = fillHoles(postprocess(scores.beat, scores.downbeat).beats, scores.permissive);
    const changes = first.length > 8 ? detectChanges(mel, first, options) : [];
    if (changes.length) {
        const cuts = [0, ...changes.map(time => round(time * FRAME_RATE)).filter(frame => frame > 50 && frame < frames - 50), frames];
        const alone: Scores = { beat: new Float32Array(frames), downbeat: new Float32Array(frames), permissive: new Float32Array(frames) };
        for (let i = 0; i < cuts.length - 1; i++) {
            const song = await overlappedScores(session, mel, cuts[i], cuts[i + 1], step, signal);
            for (const key of ['beat', 'downbeat', 'permissive'] as const) alone[key].set(song[key], cuts[i]);
        }
        scores = preferContext(scores, alone, changes);
    }
    progress('Beat This inference', total, total);
    const interleaved = new Float32Array(frames * 2);
    for (let i = 0; i < frames; i++) { interleaved[2 * i] = scores.beat[i]; interleaved[2 * i + 1] = scores.downbeat[i]; }
    return { scores: interleaved, changes, ...decodeSongs(scores, changes) };
}

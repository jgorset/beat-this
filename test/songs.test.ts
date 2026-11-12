import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mock, test } from 'node:test';
import * as ort from 'onnxruntime-web/webgpu';
import { changeProbability, detectChanges, findPeaks, maximumFilter, medianFilter, minimumFilter, uniformFilter } from '../dist/changes.js';
import { createBeatThis } from '../dist/index.js';
import { countBars, fillHoles, overlappedScores, passCount, preferContext } from '../dist/songs.js';

const logits = (frames, peaks, high = 4) => { const out = new Float32Array(frames).fill(-4); for (const time of peaks) out[Math.round(time * 50)] = high; return out; };
const every = (from, to, step) => Array.from({length:Math.round((to - from) / step)}, (_,i) => +(from + i * step).toFixed(2));

test('filters and peak picking match scipy', () => {
    assert.deepEqual([...uniformFilter([1,2,6,4],2)], [1,1.5,4,5]);
    assert.deepEqual([...maximumFilter([1,5,2,0,3],3)], [5,5,5,3,3]);
    assert.deepEqual([...minimumFilter([1,5,2,0,3],3)], [1,1,0,0,0]);
    assert.deepEqual([...medianFilter([9,1,2,8,3],3)], [9,2,2,3,3]);
    assert.deepEqual(findPeaks([0,2,0,3,3,0,1,0],0,1), [1,3,6]);
    assert.deepEqual(findPeaks([0,2,0,3,3,0,1,0],2,1), [1,3]);
    assert.deepEqual(findPeaks([0,2,0,3,0,1,0],0,3), [3]);
});

test('the song-change model gives the probabilities it was exported with', () => {
    const {inputs,outputs} = JSON.parse(readFileSync(new URL('./fixtures/changes.json', import.meta.url), 'utf8'));
    inputs.forEach((row,i) => assert.ok(Math.abs(changeProbability(row.map(value => value ?? NaN)) - outputs[i]) < 1e-12));
});

test('holes are filled from the permissive scores, and only holes', () => {
    const beats = [...every(1,3,.5), ...every(6,8,.5)], permissive = logits(500, every(1,8,.5));
    assert.deepEqual(fillHoles(beats, permissive), every(1,8,.5));
    assert.deepEqual(fillHoles(every(1,8,.5), permissive), every(1,8,.5));
    assert.deepEqual(fillHoles(beats, logits(500,[])), beats);
    assert.deepEqual(fillHoles([1], permissive), every(1,8,.5));
});

test('bars are counted steadily, across a dropped beat, and again from every song change', () => {
    const beats = every(1,13,.5), bars = every(1,13,2);
    assert.deepEqual(countBars(beats, logits(700,bars)), bars);
    // One weak bar start and one stray one do not move the count.
    assert.deepEqual(countBars(beats, logits(700,[...bars.filter(time => time !== 5), 6.5])), bars);
    // A dropped beat advances the count by two.
    assert.deepEqual(countBars(beats.filter(time => time !== 4), logits(700,bars)), bars);
    // Three in a bar is found when the evidence is there.
    assert.deepEqual(countBars(beats, logits(700,every(1,13,1.5))), every(1,13,1.5));
    // The second song starts a beat late: the first count runs to the change, and a new one starts there.
    const second = every(7.5,13,2), shifted = logits(700,[...every(1,7,2), ...second]);
    assert.deepEqual(countBars(beats, shifted, [7.25]), [...every(1,9,2), ...second]);
    assert.deepEqual(countBars([], logits(10,[])), []);
});

test('a song keeps the scores made with its neighbours only when they hear clearly more beat', () => {
    const loud = logits(1000, every(0,20,.5)), faint = logits(1000, every(0,10,.5)), scores = beat => ({beat, downbeat:beat, permissive:beat});
    const chosen = preferContext(scores(loud), scores(faint), [10]);
    assert.deepEqual(chosen.beat.subarray(0,500), faint.subarray(0,500));
    assert.deepEqual(chosen.beat.subarray(500), loud.subarray(500));
});

test('overlapping passes take every frame from the pass in which it is most central', async () => {
    const mel = new Float32Array(4000 * 128);
    for (let frame = 0; frame < 4000; frame++) mel[frame * 128] = frame;
    const starts = [];
    const session = { async run({spect}) {
        const start = spect.data[0]; starts.push(start);
        const beat = new Float32Array(1500).fill(start), downbeat = Float32Array.from({length:1500}, (_,i) => spect.data[i * 128]);
        return {beat:new ort.Tensor('float32',beat,[1,1500]), downbeat:new ort.Tensor('float32',downbeat,[1,1500])};
    } };
    const scores = await overlappedScores(session, mel);
    assert.deepEqual(starts, [0,750,1500,2250,2500]);
    assert.equal(passCount(4000), 5);
    assert.deepEqual(scores.downbeat, Float32Array.from({length:4000}, (_,frame) => frame));
    assert.deepEqual([0,749,1000,1300,1700,2400,2700,3250,3999].map(frame => scores.beat[frame]), [0,0,0,750,750,1500,2250,2500,2500]);
    const part = await overlappedScores(session, mel, 1000, 3000);
    assert.equal(part.beat.length, 2000);
    assert.equal(part.downbeat[0], 1000);
});

test('silence has no song changes, and the tracker reports changes only when asked for songs', async () => {
    assert.deepEqual(detectChanges(new Float32Array(3000 * 128), every(1,59,.5)), []);
    mock.method(ort.InferenceSession,'create',async()=>({
        async run({spect}) { const count=spect.dims[1], b=new Float32Array(count).fill(-1);
            for (let i=6;i<count-6;i+=25) b[i]=1;
            return {beat:new ort.Tensor('float32',b,[1,count]),downbeat:new ort.Tensor('float32',b.slice(),[1,count])}; },
        async release() {},
    }));
    try {
        const tracker=await createBeatThis({modelUrl:'model.onnx',executionProvider:'wasm'}), stages=new Set();
        const plain=await tracker.analyze(new Float32Array(22050*40));
        const songs=await tracker.analyze(new Float32Array(22050*40),{songs:true,onProgress:stage=>stages.add(stage)});
        assert.deepEqual(plain.changes,[]); assert.deepEqual(songs.changes,[]);
        assert.ok(songs.beats.length > 60 && songs.downbeats.length > 10);
        assert.equal(songs.beatConfidence.length, plain.beatConfidence.length);
        assert.deepEqual([...stages],['Reading audio','Beat This inference']);
        await tracker.dispose();
    } finally { mock.restoreAll(); }
});

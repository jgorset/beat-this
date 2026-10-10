import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import { extractFeatures, peakTimes, postprocess } from '../dist/index.js';

const golden = JSON.parse(readFileSync(new URL('./fixtures/preprocessing.json', import.meta.url), 'utf8'));
test('preprocessing preserves the tested port on short audio and chunk boundaries', () => {
    for (const {length, sha256} of golden) {
        const pcm = Float32Array.from({length}, (_,i) => Math.sin(i * .037) * .2 + Math.cos(i * .011) * .1);
        const actual = extractFeatures(pcm);
        assert.equal(createHash('sha256').update(new Uint8Array(actual.buffer)).digest('hex'), sha256);
        const frames = Math.floor(length / 441) + 1;
        const streamed = new Float32Array(frames * 128);
        for (let first = 0; first < frames; first += 100) {
            const count = Math.min(100, frames - first);
            const begin = Math.max(0, first * 441 - 1024), end = Math.min(length, (first + count) * 441 + 1024);
            streamed.set(extractFeatures(pcm.subarray(begin, end), first, count, length, begin), first * 128);
        }
        assert.deepEqual(streamed, actual);
    }
});

test('silence, plateaus and nearest-beat downbeat deduplication match upstream behavior', () => {
    assert.ok(extractFeatures(new Float32Array(22050)).every(value => value === 0));
    assert.deepEqual(peakTimes(Float32Array.of(1,1,1,-1,-1,-1,-1,-1)), [.01,.04]);
    assert.deepEqual(postprocess(Float32Array.of(-1,1,-1,-1,-1,-1,-1,-1,1),
        Float32Array.of(1,-1,-1,-1,1,-1,-1,-1,-1)), {beats:[.02,.16],downbeats:[.02]});
    assert.deepEqual(postprocess(Float32Array.of(-1,-1),Float32Array.of(-1,-1)), {beats:[],downbeats:[]});
});

test('invalid PCM, window bounds and nonfinite or mismatched scores are rejected', () => {
    assert.throws(() => extractFeatures(new Float32Array()), /Invalid audio/);
    assert.throws(() => extractFeatures(Float32Array.of(NaN)), /finite/);
    assert.throws(() => extractFeatures(new Float32Array(1024),0,99), /Invalid audio/);
    assert.throws(() => postprocess(Float32Array.of(1),new Float32Array()), /equal lengths/);
    assert.throws(() => peakTimes(Float32Array.of(Infinity)), /finite/);
});

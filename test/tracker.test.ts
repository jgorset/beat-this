import assert from 'node:assert/strict';
import { mock, test } from 'node:test';
import * as ort from 'onnxruntime-web/webgpu';
import { createBeatThis } from '../dist/index.js';

test('tracker trims events at the audio end, reports confidence and releases its session once', async () => {
    let released=0;
    mock.method(ort.InferenceSession,'create',async()=>({
        async run({spect}) {
            const count=spect.dims[1], b=new Float32Array(count), d=new Float32Array(count);
            b.fill(-1);d.fill(-1);
            for (let i=6;i<count-6;i+=25) { b[i]=1;d[i]=1; }
            return {beat:new ort.Tensor('float32',b,[1,count]),downbeat:new ort.Tensor('float32',d,[1,count])};
        },
        async release() { released++; },
    }));
    try {
        const tracker=await createBeatThis({modelUrl:'model.onnx',executionProvider:'wasm'});
        const result=await tracker.analyze(new Float32Array(22050));
        assert.deepEqual(result.beats,[0,.5]);
        assert.deepEqual(result.downbeats,[0,.5]);
        assert.equal(result.beatConfidence.length,51);
        assert.equal(result.beatConfidence[0],1/(1+Math.exp(-1)));
        await tracker.dispose();await tracker.dispose();assert.equal(released,1);
        await assert.rejects(tracker.analyze(new Float32Array(1)),/disposed/);
    } finally { mock.restoreAll(); }
});

test('streaming reads are bounded and a failed reader can be retried on the same tracker', async () => {
    let calls=0;
    mock.method(ort.InferenceSession,'create',async()=>({
        async run({spect}) { calls++; const count=spect.dims[1];
            return {beat:new ort.Tensor('float32',new Float32Array(count),[1,count]),downbeat:new ort.Tensor('float32',new Float32Array(count),[1,count])}; },
        async release() {},
    }));
    try {
        const tracker=await createBeatThis({modelUrl:'model.onnx'});
        await assert.rejects(tracker.analyze({sampleCount:22050,read:async()=>{throw new Error('read failed');}}),/read failed/);
        await assert.rejects(tracker.analyze({sampleCount:22050,read:async()=>Float32Array.of(0)}),/Invalid audio window/);
        const reads=[];
        await tracker.analyze({sampleCount:22050*61,read:async(first,count)=>{reads.push({first,count});return new Float32Array(count);}});
        assert.equal(calls,3); assert.equal(reads.length,3);
        assert.ok(reads.every(({count})=>count<=1500*441+2048));
        await tracker.dispose();
    } finally { mock.restoreAll(); }
});

test('tracker rejects invalid inputs and overlapping calls; cancellation allows a retry', async () => {
    mock.method(ort.InferenceSession,'create',async()=>({async release(){}}));
    try {
        const tracker=await createBeatThis({modelUrl:'model.onnx'});
        await assert.rejects(tracker.analyze(new Float32Array()),/nonempty/);
        await assert.rejects(tracker.analyze({sampleCount:NaN,read:async()=>new Float32Array()}),/nonempty/);
        let resume;
        const ready=new Promise(resolve=>{resume=resolve;});
        const controller=new AbortController();
        const pending=tracker.analyze({sampleCount:1,read:async()=>{await ready;return Float32Array.of(0);}},{signal:controller.signal});
        await assert.rejects(tracker.analyze(Float32Array.of(0)),/already analyzing/);
        await assert.rejects(tracker.dispose(),/Wait for analysis/);
        controller.abort();resume();
        await assert.rejects(pending,{name:'AbortError'});
        await assert.rejects(tracker.analyze(Float32Array.of(0),{signal:controller.signal}),{name:'AbortError'});
        await tracker.dispose();
        await assert.rejects(createBeatThis({modelUrl:''}),/model URL/);
        await assert.rejects(createBeatThis({modelUrl:'model.onnx',executionProvider:'invalid'}),/execution provider/);
    } finally { mock.restoreAll(); }
});

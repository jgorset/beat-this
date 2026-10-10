import assert from 'node:assert/strict';
import { test } from 'node:test';
import * as ort from 'onnxruntime-web/webgpu';
import { inferBeatThis } from '../dist/index.js';

test('chunk aggregation preserves the first prediction across shifted final overlaps', async () => {
    for (const frames of [1,1488,1490,1500,3001]) {
        let calls = 0, disposed = 0;
        const expected = new Float32Array(frames), written = new Set<number>();
        const session = {
            async run({spect}) {
                calls++;
                const count = spect.dims[1];
                const b = Float32Array.from({length:count},(_,i) => spect.data[i * 128] + calls * 10000);
                const d = Float32Array.from(b,v => -v);
                for (let i=6; i<count-6; i++) {
                    const frame = spect.data[i*128];
                    if (!written.has(frame)) { expected[frame] = b[i]; written.add(frame); }
                }
                return Object.fromEntries([['beat',b],['downbeat',d]].map(([name,data]) => {
                    const tensor = new ort.Tensor('float32',data,[1,count]);
                    const dispose = tensor.dispose.bind(tensor);
                    tensor.dispose = () => { disposed++; dispose(); };
                    return [name,tensor];
                }));
            },
        };
        const result = await inferBeatThis(session,frames,async(first,count) => {
            const input = new Float32Array(count*128);
            for (let i=0;i<count;i++) input[i*128] = first+i;
            return input;
        });
        assert.equal(written.size,frames);
        for (let frame=0;frame<frames;frame++) assert.equal(result.scores[frame*2],expected[frame]);
        assert.equal(disposed,calls*2);
    }
});

test('cancellation stops between windows and releases tensors from an in-flight run', async () => {
    const controller = new AbortController();
    let calls=0,disposed=0;
    const session = { async run({spect}) {
        calls++;
        const tensor = () => { const t = new ort.Tensor('float32',new Float32Array(spect.dims[1]),[1,spect.dims[1]]);
            t.dispose = () => { disposed++; }; return t; };
        controller.abort();
        return {beat:tensor(),downbeat:tensor()};
    }};
    await assert.rejects(inferBeatThis(session,3001,async(_,count) => new Float32Array(count*128),undefined,controller.signal), {name:'AbortError'});
    assert.equal(calls,1); assert.equal(disposed,2);
});

test('feature errors, model errors and invalid outputs surface without publishing results', async () => {
    await assert.rejects(inferBeatThis({run:async()=>{throw new Error('model failed');}},10,
        async(_,count)=>new Float32Array(count*128)), /model failed/);
    await assert.rejects(inferBeatThis({run:async()=>({})},10,async()=>new Float32Array()), /Invalid feature/);
    let disposed=0;
    const invalid = { data: Float32Array.of(NaN), dispose:()=>{disposed++;} };
    await assert.rejects(inferBeatThis({run:async()=>({beat:invalid,downbeat:invalid})},10,
        async(_,count)=>new Float32Array(count*128)), /Invalid Beat This model output/);
    assert.equal(disposed,2);
    await assert.rejects(inferBeatThis({},0,async()=>new Float32Array()), /Invalid feature frame/);
});

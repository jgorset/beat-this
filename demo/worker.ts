import { createBeatThis } from '../src/index.js';

self.onmessage = async ({data}: MessageEvent<{pcm:Float32Array;executionProvider:'webgpu'|'wasm'}>) => {
    let tracker: Awaited<ReturnType<typeof createBeatThis>> | undefined;
    try {
        self.postMessage({progress:'Loading model…'});
        tracker = await createBeatThis({modelUrl:'/beat-this.onnx',executionProvider:data.executionProvider});
        const result = await tracker.analyze(data.pcm,{onProgress:(stage,completed,total) => {
            self.postMessage({progress:total ? `${stage}: ${completed} / ${total}` : stage});
        }});
        self.postMessage({result});
    } catch(error) { self.postMessage({error:error instanceof Error ? error.message : 'Analysis failed.'}); }
    finally { await tracker?.dispose(); }
};

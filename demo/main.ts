const file = document.querySelector<HTMLInputElement>('#audio')!;
const analyze = document.querySelector<HTMLButtonElement>('#analyze')!;
const cancel = document.querySelector<HTMLButtonElement>('#cancel')!;
const provider = document.querySelector<HTMLSelectElement>('#provider')!;
const status = document.querySelector<HTMLParagraphElement>('#status')!;
const output = document.querySelector<HTMLPreElement>('#result')!;
let worker: Worker | undefined, generation = 0, running = false;
const controls = () => { file.disabled = running; provider.disabled = running; analyze.disabled = running || !file.files?.length; cancel.disabled = !running; };
const finish = () => { worker?.terminate(); worker = undefined; running = false; controls(); };
file.onchange = () => { controls(); output.textContent = ''; };
cancel.onclick = () => { generation++; finish(); status.textContent = 'Cancelled.'; };
document.addEventListener('pointerup', event => {
    const target = event.target;
    if (target instanceof HTMLElement && target.matches('button,input[type="file"],select')) target.blur();
});
analyze.onclick = async () => {
    const selected = file.files?.[0];
    if (!selected) return;
    const current = ++generation;
    running = true; controls(); output.textContent = ''; status.textContent = 'Decoding audio…';
    let context: AudioContext | undefined;
    try {
        context = new AudioContext({sampleRate:22050});
        const decoded = await context.decodeAudioData(await selected.arrayBuffer());
        const offline = new OfflineAudioContext(1,Math.ceil(decoded.duration*22050),22050);
        const source = offline.createBufferSource(); source.buffer = decoded; source.connect(offline.destination); source.start();
        const pcm = (await offline.startRendering()).getChannelData(0);
        if (current !== generation) return;
        worker = new Worker(new URL('./worker.ts',import.meta.url),{type:'module'});
        worker.onmessage = ({data}) => {
            if (current !== generation) return;
            if (data.progress) status.textContent = data.progress;
            else if (data.error) { status.textContent = data.error; finish(); }
            else if (data.result) {
                status.textContent = `Found ${data.result.beats.length} beats and ${data.result.downbeats.length} downbeats.`;
                output.textContent = JSON.stringify({beats:data.result.beats,downbeats:data.result.downbeats},null,2);
                finish();
            }
        };
        worker.onerror = () => { status.textContent = 'The audio worker failed.'; finish(); };
        worker.postMessage({pcm,executionProvider:provider.value},[pcm.buffer]);
    } catch(error) {
        if (current === generation) { status.textContent = error instanceof Error ? error.message : 'Analysis failed.'; finish(); }
    } finally { await context?.close(); }
};

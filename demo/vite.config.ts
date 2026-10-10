import { defineConfig } from 'vite';

export default defineConfig({
    // Prebundling relocates ONNX Runtime's module without its adjacent WASM file.
    optimizeDeps: { exclude: ['onnxruntime-web', 'onnxruntime-web/webgpu'] },
});

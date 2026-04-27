import { app } from "../../scripts/app.js";

const TARGET_CLASS = "AudioPreviewEQ";
const PANEL_HEIGHT = 270;

function normalizeUiValue(value) {
    return Array.isArray(value) ? value[0] : value;
}

function formatDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds <= 0) {
        return "0:00";
    }

    const totalSeconds = Math.round(seconds);
    const minutes = Math.floor(totalSeconds / 60);
    const remainder = totalSeconds % 60;
    return `${minutes}:${String(remainder).padStart(2, "0")}`;
}

function createPreviewPanel(node) {
    if (node.__audioPreviewPanel) {
        return node.__audioPreviewPanel;
    }

    const container = document.createElement("div");
    container.style.display = "flex";
    container.style.flexDirection = "column";
    container.style.gap = "10px";
    container.style.padding = "10px";
    container.style.background = "rgba(18, 22, 28, 0.92)";
    container.style.border = "1px solid rgba(118, 138, 164, 0.35)";
    container.style.borderRadius = "10px";
    container.style.minWidth = "280px";
    container.style.color = "#e8eef7";
    container.style.boxSizing = "border-box";

    const title = document.createElement("div");
    title.textContent = "Waveform Preview";
    title.style.fontSize = "13px";
    title.style.fontWeight = "600";
    title.style.letterSpacing = "0.02em";

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "120px";
    canvas.style.background = "linear-gradient(180deg, #16202d 0%, #0b1118 100%)";
    canvas.style.borderRadius = "8px";
    canvas.style.border = "1px solid rgba(118, 138, 164, 0.18)";

    const info = document.createElement("div");
    info.textContent = "Run the node to preview processed audio.";
    info.style.fontSize = "12px";
    info.style.color = "#aab7ca";

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload = "metadata";
    audio.style.width = "100%";

    container.append(title, canvas, info, audio);

    const widget = node.addDOMWidget("waveform_preview", "preview", container, {
        hideOnZoom: false,
        getHeight: () => PANEL_HEIGHT,
    });

    const panel = {
        audio,
        canvas,
        container,
        info,
        node,
        peaks: [],
        widget,
    };

    panel.draw = () => {
        const rect = canvas.getBoundingClientRect();
        const width = Math.max(1, Math.floor(rect.width));
        const height = Math.max(1, Math.floor(rect.height));
        const ratio = window.devicePixelRatio || 1;

        canvas.width = Math.max(1, Math.floor(width * ratio));
        canvas.height = Math.max(1, Math.floor(height * ratio));

        const ctx = canvas.getContext("2d");
        ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        ctx.clearRect(0, 0, width, height);

        ctx.fillStyle = "#0b1118";
        ctx.fillRect(0, 0, width, height);

        ctx.strokeStyle = "rgba(197, 213, 232, 0.22)";
        ctx.lineWidth = 1;
        const midY = height / 2;
        ctx.beginPath();
        ctx.moveTo(0, midY);
        ctx.lineTo(width, midY);
        ctx.stroke();

        if (!panel.peaks.length) {
            ctx.fillStyle = "#91a0b4";
            ctx.font = "12px sans-serif";
            ctx.fillText("No waveform loaded yet", 12, 22);
            return;
        }

        const step = width / panel.peaks.length;
        ctx.strokeStyle = "#6fd1ff";
        ctx.lineWidth = Math.max(1, step * 0.7);
        ctx.beginPath();

        for (let i = 0; i < panel.peaks.length; i++) {
            const [minValue, maxValue] = panel.peaks[i];
            const x = i * step + step / 2;
            const y1 = midY - maxValue * (height * 0.42);
            const y2 = midY - minValue * (height * 0.42);
            ctx.moveTo(x, y1);
            ctx.lineTo(x, y2);
        }

        ctx.stroke();
    };

    panel.update = (payload = {}) => {
        const audioUrl = normalizeUiValue(payload.audio_url);
        const waveformPeaks = normalizeUiValue(payload.waveform_peaks);
        const sampleRate = normalizeUiValue(payload.sample_rate);
        const duration = normalizeUiValue(payload.duration_sec);
        const channels = normalizeUiValue(payload.channels);

        if (audioUrl) {
            audio.src = audioUrl;
            audio.load();
        }

        panel.peaks = Array.isArray(waveformPeaks) ? waveformPeaks : [];
        info.textContent = `Processed preview • ${formatDuration(duration)} • ${sampleRate || "?"} Hz • ${channels || "?"} ch`;
        panel.draw();
    };

    const previousOnResize = node.onResize;
    node.onResize = function (...args) {
        const result = previousOnResize?.apply(this, args);
        requestAnimationFrame(() => panel.draw());
        return result;
    };

    node.__audioPreviewPanel = panel;
    node.setSize([Math.max(node.size[0], 320), Math.max(node.size[1], 420)]);
    requestAnimationFrame(() => panel.draw());
    return panel;
}

app.registerExtension({
    name: "songseparator.audio-preview-eq",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData?.name;
        if (comfyClass !== TARGET_CLASS) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            createPreviewPanel(this);
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const panel = createPreviewPanel(this);
            panel.update(message ?? {});
        };
    },
});

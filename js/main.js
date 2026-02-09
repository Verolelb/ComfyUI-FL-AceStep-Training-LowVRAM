var __defProp = Object.defineProperty;
var __defNormalProp = (obj, key, value) => key in obj ? __defProp(obj, key, { enumerable: true, configurable: true, writable: true, value }) : obj[key] = value;
var __publicField = (obj, key, value) => __defNormalProp(obj, typeof key !== "symbol" ? key + "" : key, value);
import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
const TRAINING_WIDGET_STYLES = `
  .acestep-training-widget {
    --primary: #06b6d4;
    --primary-glow: rgba(6, 182, 212, 0.4);
    --secondary: #8b5cf6;
    --success: #22c55e;
    --danger: #ef4444;
    --warning: #f59e0b;
    --bg-dark: #0f0f12;
    --bg-card: #18181b;
    --bg-elevated: #1f1f23;
    --border: #27272a;
    --border-hover: #3f3f46;
    --text-primary: #fafafa;
    --text-secondary: #a1a1aa;
    --text-muted: #71717a;

    background: var(--bg-card);
    border-radius: 12px;
    border: 1px solid var(--border);
    overflow: hidden;
    position: relative;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
    box-sizing: border-box;
    height: 100%;
    min-height: 380px;
    display: flex;
    flex-direction: column;
  }

  .acestep-training-widget * {
    box-sizing: border-box;
  }

  /* Header */
  .acestep-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border);
  }

  .acestep-title {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .acestep-badge {
    padding: 2px 8px;
    background: var(--primary);
    color: white;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 500;
  }

  .acestep-badge.idle {
    background: var(--text-muted);
  }

  .acestep-badge.training {
    background: var(--success);
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }

  /* Content */
  .acestep-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    padding: 14px;
    gap: 14px;
    overflow: hidden;
  }

  /* Stats Grid */
  .acestep-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
  }

  .acestep-stat {
    background: var(--bg-elevated);
    border-radius: 8px;
    padding: 10px;
    text-align: center;
  }

  .acestep-stat-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--primary);
  }

  .acestep-stat-label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* Progress Bar */
  .acestep-progress-section {
    background: var(--bg-elevated);
    border-radius: 8px;
    padding: 12px;
  }

  .acestep-progress-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
  }

  .acestep-progress-label {
    font-size: 11px;
    color: var(--text-secondary);
  }

  .acestep-progress-value {
    font-size: 11px;
    color: var(--text-primary);
    font-weight: 500;
  }

  .acestep-progress-bar {
    height: 6px;
    background: var(--bg-dark);
    border-radius: 3px;
    overflow: hidden;
  }

  .acestep-progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    border-radius: 3px;
    transition: width 0.3s ease;
    width: 0%;
  }

  /* Loss Chart */
  .acestep-chart-section {
    background: var(--bg-elevated);
    border-radius: 8px;
    padding: 12px;
    flex: 1;
    min-height: 100px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .acestep-chart-header {
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .acestep-chart-canvas {
    width: 100%;
    height: 100%;
    display: block;
  }

  /* Status */
  .acestep-status {
    background: var(--bg-elevated);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 11px;
    color: var(--text-secondary);
    text-align: center;
    border-left: 3px solid var(--primary);
  }

  .acestep-status.error {
    border-left-color: var(--danger);
    color: var(--danger);
  }

  .acestep-status.success {
    border-left-color: var(--success);
    color: var(--success);
  }

  /* Audio Preview */
  .acestep-audio-section {
    background: var(--bg-elevated);
    border-radius: 8px;
    padding: 12px;
  }

  .acestep-audio-header {
    font-size: 11px;
    color: var(--text-secondary);
    margin-bottom: 8px;
  }

  .acestep-audio-player {
    width: 100%;
    height: 32px;
  }

  .acestep-audio-label {
    font-size: 10px;
    color: var(--text-muted);
    margin-top: 6px;
  }
`;
class TrainingWidget {
  constructor(options) {
    __publicField(this, "node");
    __publicField(this, "container");
    __publicField(this, "element");
    // UI Elements
    __publicField(this, "badgeEl", null);
    __publicField(this, "epochValueEl", null);
    __publicField(this, "stepValueEl", null);
    __publicField(this, "lossValueEl", null);
    __publicField(this, "progressFillEl", null);
    __publicField(this, "progressLabelEl", null);
    __publicField(this, "statusEl", null);
    __publicField(this, "canvasEl", null);
    __publicField(this, "audioPlayerEl", null);
    __publicField(this, "audioLabelEl", null);
    // State
    __publicField(this, "lossHistory", []);
    __publicField(this, "isTraining", false);
    // Resize handling
    __publicField(this, "resizeObserver", null);
    __publicField(this, "resizeTimeout", null);
    this.node = options.node;
    this.container = options.container;
    this.element = document.createElement("div");
    this.element.className = "acestep-training-widget";
    this.injectStyles();
    this.createUI();
    this.container.appendChild(this.element);
  }
  injectStyles() {
    const styleId = "acestep-training-styles";
    if (!document.getElementById(styleId)) {
      const style = document.createElement("style");
      style.id = styleId;
      style.textContent = TRAINING_WIDGET_STYLES;
      document.head.appendChild(style);
    }
  }
  createUI() {
    this.element.innerHTML = `
      <div class="acestep-header">
        <div class="acestep-title">
          <span>ACE-Step Training</span>
          <span class="acestep-badge idle">Idle</span>
        </div>
      </div>

      <div class="acestep-content">
        <div class="acestep-stats">
          <div class="acestep-stat">
            <div class="acestep-stat-value" data-stat="epoch">0/0</div>
            <div class="acestep-stat-label">Epoch</div>
          </div>
          <div class="acestep-stat">
            <div class="acestep-stat-value" data-stat="step">0</div>
            <div class="acestep-stat-label">Step</div>
          </div>
          <div class="acestep-stat">
            <div class="acestep-stat-value" data-stat="loss">-</div>
            <div class="acestep-stat-label">Loss</div>
          </div>
        </div>

        <div class="acestep-progress-section">
          <div class="acestep-progress-header">
            <span class="acestep-progress-label">Training Progress</span>
            <span class="acestep-progress-value" data-progress-label>0%</span>
          </div>
          <div class="acestep-progress-bar">
            <div class="acestep-progress-fill" data-progress-fill></div>
          </div>
        </div>

        <div class="acestep-chart-section">
          <div class="acestep-chart-header">Loss History</div>
          <canvas class="acestep-chart-canvas"></canvas>
        </div>

        <div class="acestep-status">
          Ready to train
        </div>

        <div class="acestep-audio-section" style="display: none;">
          <div class="acestep-audio-header">Validation Audio</div>
          <audio class="acestep-audio-player" controls></audio>
          <div class="acestep-audio-label">Checkpoint: -</div>
        </div>
      </div>
    `;
    this.badgeEl = this.element.querySelector(".acestep-badge");
    this.epochValueEl = this.element.querySelector('[data-stat="epoch"]');
    this.stepValueEl = this.element.querySelector('[data-stat="step"]');
    this.lossValueEl = this.element.querySelector('[data-stat="loss"]');
    this.progressFillEl = this.element.querySelector("[data-progress-fill]");
    this.progressLabelEl = this.element.querySelector("[data-progress-label]");
    this.statusEl = this.element.querySelector(".acestep-status");
    this.canvasEl = this.element.querySelector(".acestep-chart-canvas");
    this.audioPlayerEl = this.element.querySelector(".acestep-audio-player");
    this.audioLabelEl = this.element.querySelector(".acestep-audio-label");
    const chartSection = this.element.querySelector(".acestep-chart-section");
    if (this.canvasEl && chartSection) {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.resizeTimeout) {
          clearTimeout(this.resizeTimeout);
        }
        this.resizeTimeout = window.setTimeout(() => {
          this.drawChart();
        }, 16);
      });
      this.resizeObserver.observe(chartSection);
    }
    this.drawChart();
  }
  updateProgress(epoch, totalEpochs, loss, step, totalSteps) {
    this.isTraining = true;
    if (this.badgeEl) {
      this.badgeEl.textContent = "Training";
      this.badgeEl.className = "acestep-badge training";
    }
    if (this.epochValueEl) {
      this.epochValueEl.textContent = `${epoch}/${totalEpochs}`;
    }
    if (this.stepValueEl) {
      this.stepValueEl.textContent = step.toString();
    }
    if (this.lossValueEl) {
      this.lossValueEl.textContent = loss.toFixed(6);
    }
    const progress = epoch / totalEpochs * 100;
    if (this.progressFillEl) {
      this.progressFillEl.style.width = `${progress}%`;
    }
    if (this.progressLabelEl) {
      this.progressLabelEl.textContent = `${progress.toFixed(1)}%`;
    }
  }
  updateLossHistory(history) {
    this.lossHistory = history;
    this.drawChart();
  }
  updateStatus(message, type = "normal") {
    if (this.statusEl) {
      this.statusEl.textContent = message;
      this.statusEl.className = "acestep-status";
      if (type !== "normal") {
        this.statusEl.classList.add(type);
      }
    }
  }
  addValidationAudio(audioBase64, checkpointPath) {
    const audioSection = this.element.querySelector(".acestep-audio-section");
    if (audioSection) {
      audioSection.style.display = "block";
    }
    if (this.audioPlayerEl) {
      this.audioPlayerEl.src = audioBase64;
    }
    if (this.audioLabelEl) {
      this.audioLabelEl.textContent = `Checkpoint: ${checkpointPath}`;
    }
  }
  onTrainingComplete(finalPath) {
    this.isTraining = false;
    if (this.badgeEl) {
      this.badgeEl.textContent = "Complete";
      this.badgeEl.className = "acestep-badge";
      this.badgeEl.style.background = "#22c55e";
    }
    this.updateStatus(`Training complete! Saved to: ${finalPath}`, "success");
  }
  reset() {
    this.lossHistory = [];
    this.isTraining = false;
    if (this.badgeEl) {
      this.badgeEl.textContent = "Training";
      this.badgeEl.className = "acestep-badge training";
    }
    if (this.epochValueEl) this.epochValueEl.textContent = "0/0";
    if (this.stepValueEl) this.stepValueEl.textContent = "0";
    if (this.lossValueEl) this.lossValueEl.textContent = "-";
    if (this.progressFillEl) this.progressFillEl.style.width = "0%";
    if (this.progressLabelEl) this.progressLabelEl.textContent = "0%";
    this.updateStatus("Starting training...");
    this.drawChart();
    const audioSection = this.element.querySelector(".acestep-audio-section");
    if (audioSection) {
      audioSection.style.display = "none";
    }
  }
  drawChart() {
    if (!this.canvasEl) return;
    const canvas = this.canvasEl;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    const w = rect.width;
    const h = rect.height;
    ctx.fillStyle = "#0f0f12";
    ctx.fillRect(0, 0, w, h);
    if (this.lossHistory.length < 2) {
      ctx.fillStyle = "#71717a";
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for training data...", w / 2, h / 2);
      return;
    }
    const padding = { top: 20, right: 20, bottom: 25, left: 50 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;
    const maxStep = Math.max(...this.lossHistory.map((d) => d.step));
    const minStep = Math.min(...this.lossHistory.map((d) => d.step));
    const maxLoss = Math.max(...this.lossHistory.map((d) => d.loss));
    const minLoss = Math.min(...this.lossHistory.map((d) => d.loss));
    const lossRange = maxLoss - minLoss || 1;
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = padding.top + chartH / 4 * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();
      const lossVal = maxLoss - lossRange / 4 * i;
      ctx.fillStyle = "#71717a";
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(lossVal.toFixed(4), padding.left - 5, y + 3);
    }
    ctx.strokeStyle = "#06b6d4";
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    this.lossHistory.forEach((point, i) => {
      const x = padding.left + (point.step - minStep) / (maxStep - minStep || 1) * chartW;
      const y = padding.top + chartH - (point.loss - minLoss) / lossRange * chartH;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, "rgba(6, 182, 212, 0.3)");
    gradient.addColorStop(1, "rgba(6, 182, 212, 0)");
    ctx.fillStyle = gradient;
    ctx.beginPath();
    this.lossHistory.forEach((point, i) => {
      const x = padding.left + (point.step - minStep) / (maxStep - minStep || 1) * chartW;
      const y = padding.top + chartH - (point.loss - minLoss) / lossRange * chartH;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    const lastPoint = this.lossHistory[this.lossHistory.length - 1];
    const lastX = padding.left + (lastPoint.step - minStep) / (maxStep - minStep || 1) * chartW;
    ctx.lineTo(lastX, padding.top + chartH);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = "#fafafa";
    ctx.font = "bold 10px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(`Loss: ${lastPoint.loss.toFixed(6)}`, padding.left + 5, padding.top + 12);
  }
  dispose() {
    if (this.resizeObserver) {
      this.resizeObserver.disconnect();
      this.resizeObserver = null;
    }
    if (this.resizeTimeout) {
      clearTimeout(this.resizeTimeout);
      this.resizeTimeout = null;
    }
    if (this.element.parentNode) {
      this.element.parentNode.removeChild(this.element);
    }
  }
}
const widgetInstances = /* @__PURE__ */ new Map();
function createTrainingWidget(node) {
  const container = document.createElement("div");
  container.id = `acestep-training-widget-${node.id}`;
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.minHeight = "280px";
  const widget = node.addDOMWidget(
    "training_ui",
    "training-widget",
    container,
    {
      getMinHeight: () => 400,
      hideOnZoom: false,
      serialize: false
    }
  );
  setTimeout(() => {
    const trainingWidget = new TrainingWidget({
      node,
      container
    });
    widgetInstances.set(node.id, trainingWidget);
  }, 100);
  widget.onRemove = () => {
    const instance = widgetInstances.get(node.id);
    if (instance) {
      instance.dispose();
      widgetInstances.delete(node.id);
    }
  };
  return { widget };
}
app.registerExtension({
  name: "ComfyUI.FL_AceStep_Training",
  // Called when any node is created
  nodeCreated(node) {
    var _a;
    var comfyClass = ((_a = node.constructor) == null ? void 0 : _a.comfyClass) || "";
    // Apply FL_ theming to all FL_AceStep nodes
    if (comfyClass.startsWith("FL_")) {
      node.color = "#16727c";
      node.bgcolor = "#4F0074";
    }
    // Attach training widget only to the Train node
    if (comfyClass !== "FL_AceStep_Train") {
      return;
    }
    const [oldWidth, oldHeight] = node.size;
    node.setSize([Math.max(oldWidth, 400), Math.max(oldHeight, 600)]);
    createTrainingWidget(node);
  }
});
api.addEventListener("acestep.training.progress", ((event) => {
  const detail = event.detail;
  if (!(detail == null ? void 0 : detail.node)) return;
  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (!widget) return;
  switch (detail.type) {
    case "progress":
      widget.updateProgress(
        detail.epoch ?? 0,
        detail.total_epochs ?? 0,
        detail.loss ?? 0,
        detail.step ?? 0
      );
      if (detail.loss_history) {
        widget.updateLossHistory(detail.loss_history);
      }
      break;
    case "status":
      widget.updateStatus(detail.message ?? "");
      break;
    case "checkpoint":
      if (detail.checkpoint_path) {
        widget.updateStatus(`Checkpoint saved: ${detail.checkpoint_path}`, "success");
      }
      break;
    case "complete":
      if (detail.final_path) {
        widget.onTrainingComplete(detail.final_path);
      }
      break;
  }
}));
api.addEventListener("executed", ((event) => {
  var _a;
  const detail = event.detail;
  if (!(detail == null ? void 0 : detail.node) || !(detail == null ? void 0 : detail.output)) return;
  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (!widget) return;
  const finalPath = (_a = detail.output) == null ? void 0 : _a.final_lora_path;
  if (finalPath && finalPath.length > 0) {
    widget.onTrainingComplete(finalPath[0]);
  }
}));
api.addEventListener("executing", ((event) => {
  const detail = event.detail;
  if (!(detail == null ? void 0 : detail.node)) return;
  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (widget) {
    widget.reset();
  }
}));
export {
  TrainingWidget
};
//# sourceMappingURL=main.js.map

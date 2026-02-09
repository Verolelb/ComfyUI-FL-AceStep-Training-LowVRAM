/**
 * Training Widget Component for ACE-Step Training
 */

import { TRAINING_WIDGET_STYLES } from './styles';
import type { TrainingNode, LossPoint } from './types';

export class TrainingWidget {
  private node: TrainingNode;
  private container: HTMLElement;
  private element: HTMLElement;

  // UI Elements
  private badgeEl: HTMLElement | null = null;
  private epochValueEl: HTMLElement | null = null;
  private stepValueEl: HTMLElement | null = null;
  private lossValueEl: HTMLElement | null = null;
  private progressFillEl: HTMLElement | null = null;
  private progressLabelEl: HTMLElement | null = null;
  private statusEl: HTMLElement | null = null;
  private canvasEl: HTMLCanvasElement | null = null;
  private audioPlayerEl: HTMLAudioElement | null = null;
  private audioLabelEl: HTMLElement | null = null;

  // State
  private lossHistory: LossPoint[] = [];
  private isTraining: boolean = false;

  // Resize handling
  private resizeObserver: ResizeObserver | null = null;
  private resizeTimeout: number | null = null;

  constructor(options: { node: TrainingNode; container: HTMLElement }) {
    this.node = options.node;
    this.container = options.container;
    this.element = document.createElement('div');
    this.element.className = 'acestep-training-widget';

    this.injectStyles();
    this.createUI();
    this.container.appendChild(this.element);
  }

  private injectStyles(): void {
    const styleId = 'acestep-training-styles';
    if (!document.getElementById(styleId)) {
      const style = document.createElement('style');
      style.id = styleId;
      style.textContent = TRAINING_WIDGET_STYLES;
      document.head.appendChild(style);
    }
  }

  private createUI(): void {
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

    // Get references
    this.badgeEl = this.element.querySelector('.acestep-badge');
    this.epochValueEl = this.element.querySelector('[data-stat="epoch"]');
    this.stepValueEl = this.element.querySelector('[data-stat="step"]');
    this.lossValueEl = this.element.querySelector('[data-stat="loss"]');
    this.progressFillEl = this.element.querySelector('[data-progress-fill]');
    this.progressLabelEl = this.element.querySelector('[data-progress-label]');
    this.statusEl = this.element.querySelector('.acestep-status');
    this.canvasEl = this.element.querySelector('.acestep-chart-canvas');
    this.audioPlayerEl = this.element.querySelector('.acestep-audio-player');
    this.audioLabelEl = this.element.querySelector('.acestep-audio-label');

    // Set up ResizeObserver for canvas scaling - watch container, not canvas
    const chartSection = this.element.querySelector('.acestep-chart-section');
    if (this.canvasEl && chartSection) {
      this.resizeObserver = new ResizeObserver(() => {
        if (this.resizeTimeout) {
          clearTimeout(this.resizeTimeout);
        }
        this.resizeTimeout = window.setTimeout(() => {
          this.drawChart();
        }, 16); // ~60fps max
      });
      this.resizeObserver.observe(chartSection);
    }

    // Draw initial chart
    this.drawChart();
  }

  public updateProgress(
    epoch: number,
    totalEpochs: number,
    loss: number,
    step: number,
    totalSteps?: number
  ): void {
    this.isTraining = true;

    // Update badge
    if (this.badgeEl) {
      this.badgeEl.textContent = 'Training';
      this.badgeEl.className = 'acestep-badge training';
    }

    // Update stats
    if (this.epochValueEl) {
      this.epochValueEl.textContent = `${epoch}/${totalEpochs}`;
    }
    if (this.stepValueEl) {
      this.stepValueEl.textContent = step.toString();
    }
    if (this.lossValueEl) {
      this.lossValueEl.textContent = loss.toFixed(6);
    }

    // Update progress bar
    const progress = (epoch / totalEpochs) * 100;
    if (this.progressFillEl) {
      this.progressFillEl.style.width = `${progress}%`;
    }
    if (this.progressLabelEl) {
      this.progressLabelEl.textContent = `${progress.toFixed(1)}%`;
    }
  }

  public updateLossHistory(history: LossPoint[]): void {
    this.lossHistory = history;
    this.drawChart();
  }

  public updateStatus(message: string, type: 'normal' | 'error' | 'success' = 'normal'): void {
    if (this.statusEl) {
      this.statusEl.textContent = message;
      this.statusEl.className = 'acestep-status';
      if (type !== 'normal') {
        this.statusEl.classList.add(type);
      }
    }
  }

  public addValidationAudio(audioBase64: string, checkpointPath: string): void {
    const audioSection = this.element.querySelector('.acestep-audio-section') as HTMLElement;
    if (audioSection) {
      audioSection.style.display = 'block';
    }

    if (this.audioPlayerEl) {
      this.audioPlayerEl.src = audioBase64;
    }

    if (this.audioLabelEl) {
      this.audioLabelEl.textContent = `Checkpoint: ${checkpointPath}`;
    }
  }

  public onTrainingComplete(finalPath: string): void {
    this.isTraining = false;

    if (this.badgeEl) {
      this.badgeEl.textContent = 'Complete';
      this.badgeEl.className = 'acestep-badge';
      this.badgeEl.style.background = '#22c55e';
    }

    this.updateStatus(`Training complete! Saved to: ${finalPath}`, 'success');
  }

  public reset(): void {
    this.lossHistory = [];
    this.isTraining = false;

    if (this.badgeEl) {
      this.badgeEl.textContent = 'Training';
      this.badgeEl.className = 'acestep-badge training';
    }

    if (this.epochValueEl) this.epochValueEl.textContent = '0/0';
    if (this.stepValueEl) this.stepValueEl.textContent = '0';
    if (this.lossValueEl) this.lossValueEl.textContent = '-';

    if (this.progressFillEl) this.progressFillEl.style.width = '0%';
    if (this.progressLabelEl) this.progressLabelEl.textContent = '0%';

    this.updateStatus('Starting training...');
    this.drawChart();

    // Hide audio section
    const audioSection = this.element.querySelector('.acestep-audio-section') as HTMLElement;
    if (audioSection) {
      audioSection.style.display = 'none';
    }
  }

  private drawChart(): void {
    if (!this.canvasEl) return;

    const canvas = this.canvasEl;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Get actual size
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const w = rect.width;
    const h = rect.height;

    // Clear
    ctx.fillStyle = '#0f0f12';
    ctx.fillRect(0, 0, w, h);

    // No data message
    if (this.lossHistory.length < 2) {
      ctx.fillStyle = '#71717a';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for training data...', w / 2, h / 2);
      return;
    }

    // Calculate scales
    const padding = { top: 20, right: 20, bottom: 25, left: 50 };
    const chartW = w - padding.left - padding.right;
    const chartH = h - padding.top - padding.bottom;

    const maxStep = Math.max(...this.lossHistory.map(d => d.step));
    const minStep = Math.min(...this.lossHistory.map(d => d.step));
    const maxLoss = Math.max(...this.lossHistory.map(d => d.loss));
    const minLoss = Math.min(...this.lossHistory.map(d => d.loss));
    const lossRange = maxLoss - minLoss || 1;

    // Draw grid
    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 0.5;

    for (let i = 0; i <= 4; i++) {
      const y = padding.top + (chartH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(w - padding.right, y);
      ctx.stroke();

      // Y-axis labels
      const lossVal = maxLoss - (lossRange / 4) * i;
      ctx.fillStyle = '#71717a';
      ctx.font = '9px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(lossVal.toFixed(4), padding.left - 5, y + 3);
    }

    // Draw loss line
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 2;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    ctx.beginPath();

    this.lossHistory.forEach((point, i) => {
      const x = padding.left + ((point.step - minStep) / (maxStep - minStep || 1)) * chartW;
      const y = padding.top + chartH - ((point.loss - minLoss) / lossRange) * chartH;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // Draw gradient fill
    const gradient = ctx.createLinearGradient(0, padding.top, 0, h - padding.bottom);
    gradient.addColorStop(0, 'rgba(6, 182, 212, 0.3)');
    gradient.addColorStop(1, 'rgba(6, 182, 212, 0)');

    ctx.fillStyle = gradient;
    ctx.beginPath();

    this.lossHistory.forEach((point, i) => {
      const x = padding.left + ((point.step - minStep) / (maxStep - minStep || 1)) * chartW;
      const y = padding.top + chartH - ((point.loss - minLoss) / lossRange) * chartH;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    // Close the path
    const lastPoint = this.lossHistory[this.lossHistory.length - 1];
    const lastX = padding.left + ((lastPoint.step - minStep) / (maxStep - minStep || 1)) * chartW;
    ctx.lineTo(lastX, padding.top + chartH);
    ctx.lineTo(padding.left, padding.top + chartH);
    ctx.closePath();
    ctx.fill();

    // Current loss label
    ctx.fillStyle = '#fafafa';
    ctx.font = 'bold 10px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${lastPoint.loss.toFixed(6)}`, padding.left + 5, padding.top + 12);
  }

  public dispose(): void {
    // Clean up ResizeObserver
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

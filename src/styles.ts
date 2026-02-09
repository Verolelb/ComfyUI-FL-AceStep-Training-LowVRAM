/**
 * CSS styles for ACE-Step Training widget
 */

export const TRAINING_WIDGET_STYLES = `
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

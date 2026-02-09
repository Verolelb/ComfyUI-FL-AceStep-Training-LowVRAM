/**
 * ACE-Step Training Frontend Extension
 *
 * Registers the training widget with ComfyUI and handles WebSocket events.
 */

import { app } from '../../../scripts/app.js';
import { api } from '../../../scripts/api.js';
import { TrainingWidget } from './TrainingWidget';
import type { TrainingNode, DOMWidget, TrainingProgressData } from './types';

// Store widget instances for cleanup and updates
const widgetInstances = new Map<number, TrainingWidget>();

function createTrainingWidget(node: TrainingNode): { widget: DOMWidget } {
  // Create container element
  const container = document.createElement('div');
  container.id = `acestep-training-widget-${node.id}`;
  container.style.width = '100%';
  container.style.height = '100%';
  container.style.minHeight = '280px';

  // Create DOM widget using ComfyUI's API
  const widget = node.addDOMWidget(
    'training_ui',
    'training-widget',
    container,
    {
      getMinHeight: () => 400,
      hideOnZoom: false,
      serialize: false,
    }
  ) as DOMWidget;

  // Create the actual widget after container is mounted
  setTimeout(() => {
    const trainingWidget = new TrainingWidget({
      node,
      container,
    });
    widgetInstances.set(node.id, trainingWidget);
  }, 100);

  // Cleanup when node is removed
  widget.onRemove = () => {
    const instance = widgetInstances.get(node.id);
    if (instance) {
      instance.dispose();
      widgetInstances.delete(node.id);
    }
  };

  return { widget };
}

// Register the extension with ComfyUI
app.registerExtension({
  name: 'ComfyUI.FL_AceStep_Training',

  // Called when any node is created
  nodeCreated(node: TrainingNode) {
    const comfyClass = node.constructor?.comfyClass || '';

    // Apply FL_ theming to all FL_AceStep nodes
    if (comfyClass.startsWith('FL_')) {
      node.color = '#16727c';
      node.bgcolor = '#4F0074';
    }

    // Attach training widget only to the Train node
    if (comfyClass !== 'FL_AceStep_Train') {
      return;
    }

    // Adjust default node size to accommodate the widget
    const [oldWidth, oldHeight] = node.size;
    node.setSize([Math.max(oldWidth, 400), Math.max(oldHeight, 600)]);

    // Create the training widget
    createTrainingWidget(node);
  },
});

// Listen for custom training updates from Python
api.addEventListener('acestep.training.progress', ((event: CustomEvent<TrainingProgressData>) => {
  const detail = event.detail;
  if (!detail?.node) return;

  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (!widget) return;

  switch (detail.type) {
    case 'progress':
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

    case 'status':
      widget.updateStatus(detail.message ?? '');
      break;

    case 'checkpoint':
      if (detail.checkpoint_path) {
        widget.updateStatus(`Checkpoint saved: ${detail.checkpoint_path}`, 'success');
      }
      break;

    case 'complete':
      if (detail.final_path) {
        widget.onTrainingComplete(detail.final_path);
      }
      break;
  }
}) as EventListener);

// Listen for node execution results
api.addEventListener('executed', ((event: CustomEvent) => {
  const detail = event.detail;
  if (!detail?.node || !detail?.output) return;

  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (!widget) return;

  // Access the ui data sent from Python
  const finalPath = detail.output?.final_lora_path as string[] | undefined;

  if (finalPath && finalPath.length > 0) {
    widget.onTrainingComplete(finalPath[0]);
  }
}) as EventListener);

// Listen for execution start to reset the widget
api.addEventListener('executing', ((event: CustomEvent) => {
  const detail = event.detail;
  if (!detail?.node) return;

  const nodeId = parseInt(detail.node, 10);
  const widget = widgetInstances.get(nodeId);
  if (widget) {
    widget.reset();
  }
}) as EventListener);

export { TrainingWidget };

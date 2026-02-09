/**
 * TypeScript type definitions for ACE-Step Training widget
 */

export interface TrainingProgressData {
  type: 'progress' | 'status' | 'checkpoint' | 'complete';
  node: string;
  epoch?: number;
  total_epochs?: number;
  step?: number;
  loss?: number;
  lr?: number;
  loss_history?: LossPoint[];
  message?: string;
  checkpoint_path?: string;
  final_path?: string;
}

export interface LossPoint {
  step: number;
  loss: number;
}

export interface TrainingNode {
  id: number;
  size: [number, number];
  setSize: (size: [number, number]) => void;
  addDOMWidget: (
    name: string,
    type: string,
    element: HTMLElement,
    options?: DOMWidgetOptions
  ) => DOMWidget;
  constructor?: {
    comfyClass?: string;
  };
}

export interface DOMWidgetOptions {
  getMinHeight?: () => number;
  hideOnZoom?: boolean;
  serialize?: boolean;
}

export interface DOMWidget {
  element: HTMLElement;
  onRemove?: () => void;
}

export interface TrainingWidgetOptions {
  node: TrainingNode;
  container: HTMLElement;
}

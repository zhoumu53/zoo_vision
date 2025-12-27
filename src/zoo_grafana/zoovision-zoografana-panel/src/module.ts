import { PanelPlugin } from '@grafana/data';
import { ZooTracksOptions } from './types';
import { ZooTracksPanel } from './components/ZooTracksPanel';

export const plugin = new PanelPlugin<ZooTracksOptions>(ZooTracksPanel).setPanelOptions((builder) => {
  return builder;
});

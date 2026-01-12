import { PanelPlugin } from '@grafana/data';
import { ZooTracksOptions } from './types';
import { ZooTracksPanel } from './components/ZooTracksPanel';

export const plugin = new PanelPlugin<ZooTracksOptions>(ZooTracksPanel).setPanelOptions((builder) => {
  return builder
    .addTextInput({
      path: 'track_images_server',
      name: 'Track images server',
      description: 'The url of the server serving the images. E.g. http://127.0.0.1:5000',
      defaultValue: 'http://127.0.0.1:5000',
    });
});

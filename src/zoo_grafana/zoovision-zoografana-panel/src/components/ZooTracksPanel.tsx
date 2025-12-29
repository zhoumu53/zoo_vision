import React, { useEffect, useState } from 'react';
import { PanelProps, DataHoverEvent } from '@grafana/data';
import { ZooTracksOptions } from 'types';
import { css, cx } from '@emotion/css';
import { useStyles2 } from '@grafana/ui';
import { PanelDataErrorView } from '@grafana/runtime';

const ZOO_DASHBOARD_SERVER = "127.0.0.1:5000";
const DEFAULT_TIMESTAMP = 1742096040000;
const CAMERAS = ["zag_elp_cam_016", "zag_elp_cam_017", "zag_elp_cam_018", "zag_elp_cam_019"];

interface Props extends PanelProps<ZooTracksOptions> { }

const getStyles = () => {
  return {
    wrapper: css`
      font-family: Open Sans;
      position: relative;
    `,
    svg: css`
      position: absolute;
      top: 0;
      left: 0;
    `,
    textBox: css`
      position: absolute;
      bottom: 0;
      left: 0;
      padding: 10px;
    `,
    areaName: css`
      padding: 10px;
    `,
    rowFlex: css`
      display: flex;
      flex-flow: row;
    `,
    trackImage: css`
      max-width: 100%;
      max-height: 100%;
    `,

  };
};

function buildTrackImagesUrl(cameraName: string, timestamp: number): string {
  return `http://${ZOO_DASHBOARD_SERVER}/track_images?camera=${cameraName}&timestamp=${timestamp}`;
}

export const ZooTracksPanel: React.FC<Props> = ({ eventBus, options, data, width, height, fieldConfig, id }) => {
  const styles = useStyles2(getStyles);

  const [currentTimestamp, setCurrentTimestamp] = useState<number>(DEFAULT_TIMESTAMP);
  const [imagesCamera0, imagesSetState0] = useState<string>(buildTrackImagesUrl(CAMERAS[0], DEFAULT_TIMESTAMP));
  const [imagesCamera1, imagesSetState1] = useState<string>(buildTrackImagesUrl(CAMERAS[1], DEFAULT_TIMESTAMP));
  const [imagesCamera2, imagesSetState2] = useState<string>(buildTrackImagesUrl(CAMERAS[2], DEFAULT_TIMESTAMP));
  const [imagesCamera3, imagesSetState3] = useState<string>(buildTrackImagesUrl(CAMERAS[3], DEFAULT_TIMESTAMP));
  const cameraImages = [imagesCamera0, imagesCamera1, imagesCamera2, imagesCamera3];
  const cameraSetState = [imagesSetState0, imagesSetState1, imagesSetState2, imagesSetState3];

  useEffect(() => {
    const subscriber = eventBus.getStream(DataHoverEvent).subscribe((event) => {
      let timestamp = event.payload.point.time;
      if (timestamp == null) {
        return;
      }

      setCurrentTimestamp(timestamp);

      for (const index in CAMERAS) {
        cameraSetState[index](buildTrackImagesUrl(CAMERAS[index], timestamp));
      }
    });

    return () => {
      subscriber.unsubscribe();
    };
  });

  if (data.series.length === 0) {
    return <PanelDataErrorView fieldConfig={fieldConfig} panelId={id} data={data} needsStringField />;
  }

  return (
    <div
      className={cx(
        styles.wrapper,
        css`
          width: ${width}px;
          height: ${height}px;
        `
      )}
    >
      <div id="time-label">Time: {new Date(currentTimestamp).toLocaleString()}</div>
      <div className={cx(styles.rowFlex)}>
        <div className={cx(styles.areaName)}>
          <h2>
            Sand box mit
          </h2>
          <div className={cx(styles.rowFlex)}>
            <div>
              <div>Cam017</div>
              <img className={cx(styles.trackImage)} src={cameraImages[1]} />
            </div>
            <div>
              <div>Cam018</div>
              <img className={cx(styles.trackImage)} src={cameraImages[2]} />
            </div>
          </div>
        </div>
        <div className={cx(styles.areaName)}>
          <h2>
            Sand box ohne
          </h2>
          <div className={cx(styles.rowFlex)}>
            <div>
              <div>Cam016</div>
              <img className={cx(styles.trackImage)} src={cameraImages[0]} />
            </div>
            <div>
              <div>Cam019</div>
              <img className={cx(styles.trackImage)} src={cameraImages[3]} />
            </div>
          </div>
        </div>
      </div>
    </div >
  );
};

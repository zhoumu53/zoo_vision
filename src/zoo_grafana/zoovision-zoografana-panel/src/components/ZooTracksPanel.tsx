import React, { useEffect, useState } from 'react';
import { PanelProps, DataHoverEvent } from '@grafana/data';
import { ZooTracksOptions } from 'types';
import { css, cx } from '@emotion/css';
import { useStyles2 } from '@grafana/ui';

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
      width: 50%;
    `,
    rowFlex: css`
      display: flex;
      flex-flow: row;
    `,
    trackImage: css`
      max-width: 100%;
      max-height: 100%;
      border-style: solid;
      border-width: thin;
      border-color: gray;
    `,

  };
};

function buildTrackImagesUrl(cameraName: string, timestamp: number): string {
  return `http://${ZOO_DASHBOARD_SERVER}/track_images?camera=${cameraName}&timestamp=${timestamp}`;
}

async function fetchImages(cameraName: string, timestamp: number, setState: (state: any) => void) {
  const data = await fetch(buildTrackImagesUrl(cameraName, timestamp));
  const dataJson = await data.json();
  setState(dataJson["result"])
}

async function changeTimestamp(timestamp: number, setCurrentTimestamp: any, cameraSetState: any) {
  setCurrentTimestamp(timestamp);

  for (const index in CAMERAS) {
    fetchImages(CAMERAS[index], timestamp, cameraSetState[index]);
  }
}

export const ZooTracksPanel: React.FC<Props> = ({ eventBus, options, data, width, height, fieldConfig, id }) => {
  const styles = useStyles2(getStyles);

  const [currentTimestamp, setCurrentTimestamp] = useState<number>(0);
  const [imagesCamera0, imagesSetState0] = useState<string[]>([]);
  const [imagesCamera1, imagesSetState1] = useState<string[]>([]);
  const [imagesCamera2, imagesSetState2] = useState<string[]>([]);
  const [imagesCamera3, imagesSetState3] = useState<string[]>([]);
  const cameraImages = [imagesCamera0, imagesCamera1, imagesCamera2, imagesCamera3];
  const cameraSetState = [imagesSetState0, imagesSetState1, imagesSetState2, imagesSetState3];

  if (currentTimestamp === 0) {
    changeTimestamp(DEFAULT_TIMESTAMP, setCurrentTimestamp, cameraSetState);
  }

  useEffect(() => {
    const subscriber = eventBus.getStream(DataHoverEvent).subscribe((event) => {
      let timestamp = event.payload.point.time;
      if (timestamp == null) {
        return;
      }
      changeTimestamp(timestamp, setCurrentTimestamp, cameraSetState);
    });

    return () => {
      subscriber.unsubscribe();
    };
  });

  const makeImages = (index: number) => {
    return <div>
      <div>{CAMERAS[index]}</div>
      {
        cameraImages[index].map((value, index, _arr) =>
          <img key={index} className={cx(styles.trackImage)} src={`data:image/jpeg;base64,${value}`} />)
      }
    </div>

  };

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
            {makeImages(1)}
            {makeImages(2)}
          </div>
        </div>
        <div className={cx(styles.areaName)}>
          <h2>
            Sand box ohne
          </h2>
          <div className={cx(styles.rowFlex)}>
            {makeImages(0)}
            {makeImages(3)}
          </div>
        </div>
      </div>
    </div >
  );
};

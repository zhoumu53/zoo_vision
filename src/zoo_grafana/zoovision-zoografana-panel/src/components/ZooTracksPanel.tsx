import React, { useEffect, useState } from 'react';
import { PanelProps, DataHoverEvent } from '@grafana/data';
import { ZooTracksOptions } from 'types';
import { css, cx } from '@emotion/css';
import { useStyles2 } from '@grafana/ui';


const ZOO_DASHBOARD_SERVER = "127.0.0.1:5000";
const DEFAULT_TIMESTAMP_MS = 1742096040000;
const CAMERAS = ["zag_elp_cam_016", "zag_elp_cam_017", "zag_elp_cam_018", "zag_elp_cam_019"];

interface Props extends PanelProps<ZooTracksOptions> { }

const getStyles = () => {
  return {
    wrapper: css`
      font-family: Open Sans;
      position: relative;
    `,
    areaName: css`
      padding: 10px;
      width: 50%;
    `,
    rowFlex: css`
      display: flex;
      flex-flow: row;
      width: 100%;
    `,
    cameraBlock: css`
      width: 50%;
    `,
    trackImageContainer: css`
      width: 100%;
      display: flex;
      flex-direction: row;
      flex-wrap: wrap;
      padding: 1px;
    `,
    trackImageDiv: css`
      max-width: 50%;
      border-style: solid;
      border-width: thin;
      border-color: gray;
    `,
    trackImage: css`
      width: 100%;
      height: 100%;
    `,

  };
};

function buildTrackImagesUrl(cameraName: string, timestamp_ms: number): string {
  const timestamp_s = timestamp_ms / 1000;
  const utcOffset_s = new Date().getTimezoneOffset() * 60;
  const timestampUtc_s = timestamp_s + utcOffset_s;
  return `http://${ZOO_DASHBOARD_SERVER}/track_images?camera=${cameraName}&timestamp=${timestampUtc_s}`;
}

async function fetchImages(cameraName: string, timestamp_ms: number, setCurrentTimestamp: any, setState: (state: any) => void) {
  const data = await fetch(buildTrackImagesUrl(cameraName, timestamp_ms));
  const dataJson = await data.json();
  setCurrentTimestamp(dataJson["timestamp"])
  setState(dataJson["images"])
}

async function changeTimestamp(timestamp_ms: number, setCurrentTimestamp: any, cameraSetState: any) {
  for (const index in CAMERAS) {
    fetchImages(CAMERAS[index], timestamp_ms, setCurrentTimestamp, cameraSetState[index]);
  }
}

export const ZooTracksPanel: React.FC<Props> = ({ eventBus, options, data, width, height, fieldConfig, id }) => {
  const styles = useStyles2(getStyles);

  const [currentTimestamp, setCurrentTimestamp] = useState<string>("");
  const [imagesCamera0, imagesSetState0] = useState<string[]>([]);
  const [imagesCamera1, imagesSetState1] = useState<string[]>([]);
  const [imagesCamera2, imagesSetState2] = useState<string[]>([]);
  const [imagesCamera3, imagesSetState3] = useState<string[]>([]);
  const cameraImages = [imagesCamera0, imagesCamera1, imagesCamera2, imagesCamera3];
  const cameraSetState = [imagesSetState0, imagesSetState1, imagesSetState2, imagesSetState3];

  if (currentTimestamp === "") {
    changeTimestamp(DEFAULT_TIMESTAMP_MS, setCurrentTimestamp, cameraSetState);
  }

  useEffect(() => {
    const subscriber = eventBus.getStream(DataHoverEvent).subscribe((event) => {
      if (event.payload.point == null) {
        return;
      }
      let timestamp_ms = event.payload.point.time;
      if (timestamp_ms == null) {
        return;
      }
      changeTimestamp(timestamp_ms, setCurrentTimestamp, cameraSetState);
    });

    return () => {
      subscriber.unsubscribe();
    };
  });

  const makeImages = (cameraIndex: number) => {
    return <div className={cx(styles.cameraBlock)}>
      <div>{CAMERAS[cameraIndex]}</div>
      <div className={cx(styles.trackImageContainer)}>
        {cameraImages[cameraIndex].map((value, index) =>
          <div className={cx(styles.trackImageDiv)}>
            <img key={index} src={`data:image/jpeg;base64,${value}`} className={cx(styles.trackImage)} />
          </div>
        )}
      </div>
    </div>
  };

  return (
    <div
      className={cx(
        styles.wrapper,
        css`
          width: ${width}px;
          height: ${height}px;
          overflow: auto;
        `
      )}
    >
      <div id="time-label">Time: {currentTimestamp}</div>
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

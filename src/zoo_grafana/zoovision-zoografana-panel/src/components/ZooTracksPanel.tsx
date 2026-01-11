import React, { useEffect, useState } from 'react';
import { PanelProps, DataHoverEvent } from '@grafana/data';
import { ZooTracksOptions } from 'types';
import { css, cx } from '@emotion/css';
import { useStyles2 } from '@grafana/ui';

const NO_IMAGE_JPG = require('../img/no_image.jpg');

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
    fillImage: css`
      width: 100%;
    `,

  };
};

function buildTrackImagesUrl(track_images_server: string, cameraName: string, timestamp_ms: number): string {
  const timestamp_s = timestamp_ms / 1000;
  const utcOffset_s = new Date().getTimezoneOffset() * 60;
  const timestampUtc_s = timestamp_s + utcOffset_s;
  return `${track_images_server}/track_images?camera=${cameraName}&timestamp=${timestampUtc_s}`;
}

function buildCameraImageUrl(track_images_server: string, cameraName: string, timestamp_ms: number): string {
  const timestamp_s = timestamp_ms / 1000;
  const utcOffset_s = new Date().getTimezoneOffset() * 60;
  const timestampUtc_s = timestamp_s + utcOffset_s;
  return `${track_images_server}/camera_image?camera=${cameraName}&timestamp=${timestampUtc_s}`;
}

async function fetchImages(url: string, setCurrentTimestamp: any, setState: (state: any) => void) {
  let image_src = [""];
  let server_timestamp = "";
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const dataJson = await response.json();
    server_timestamp = dataJson["timestamp"];
    image_src = dataJson["images"].map((jpg_data: string) => `data:image/jpeg;base64,${jpg_data}`);
  } catch (e) {
    console.log("Error fetching %s: %s", url, e);
    server_timestamp = "n/a"
    image_src = [NO_IMAGE_JPG];
  }
  setCurrentTimestamp(server_timestamp)
  setState(image_src);
}

async function changeTimestamp(track_images_server: string, timestamp_ms: number, setCurrentTimestamp: any, cameraSetState: any, cameraImagesSetState: any) {
  for (const index in CAMERAS) {
    const url = buildTrackImagesUrl(track_images_server, CAMERAS[index], timestamp_ms);
    fetchImages(url, setCurrentTimestamp, cameraSetState[index]);
    cameraImagesSetState[index](buildCameraImageUrl(track_images_server, CAMERAS[index], timestamp_ms));
  }
}

export const ZooTracksPanel: React.FC<Props> = ({ eventBus, options, data, width, height, fieldConfig, id }) => {
  const styles = useStyles2(getStyles);

  const [currentTimestamp, setCurrentTimestamp] = useState<string>("DEFAULT");
  const [trackImages0, trackImagesSetState0] = useState<string[]>([]);
  const [trackImages1, trackImagesSetState1] = useState<string[]>([]);
  const [trackImages2, trackImagesSetState2] = useState<string[]>([]);
  const [trackImages3, trackImagesSetState3] = useState<string[]>([]);
  const trackImages = [trackImages0, trackImages1, trackImages2, trackImages3];
  const trackImagesSetState = [trackImagesSetState0, trackImagesSetState1, trackImagesSetState2, trackImagesSetState3];

  const [cameraImage0, cameraImageSetState0] = useState<string>("");
  const [cameraImage1, cameraImageSetState1] = useState<string>("");
  const [cameraImage2, cameraImageSetState2] = useState<string>("");
  const [cameraImage3, cameraImageSetState3] = useState<string>("");
  const cameraImages = [cameraImage0, cameraImage1, cameraImage2, cameraImage3];
  const cameraImagesSetState = [cameraImageSetState0, cameraImageSetState1, cameraImageSetState2, cameraImageSetState3];

  if (currentTimestamp === "DEFAULT") {
    setCurrentTimestamp("");
    changeTimestamp(options.track_images_server, DEFAULT_TIMESTAMP_MS, setCurrentTimestamp, trackImagesSetState, cameraImagesSetState);
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
      changeTimestamp(options.track_images_server, timestamp_ms, setCurrentTimestamp, trackImagesSetState, cameraImagesSetState);
    });

    return () => {
      subscriber.unsubscribe();
    };
  });

  const makeImages = (cameraIndex: number) => {
    return <div className={cx(styles.cameraBlock)}>
      <div>{CAMERAS[cameraIndex]}</div>
      <div>Tracks</div>
      <div className={cx(styles.trackImageContainer)}>
        {trackImages[cameraIndex].map((value, index) =>
          <div key={index} className={cx(styles.trackImageDiv)}>
            <img src={value} className={cx(styles.fillImage)} />
          </div>
        )}
      </div>
      <div>Source</div>
      <img src={cameraImages[cameraIndex]} className={cx(styles.fillImage)} />
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

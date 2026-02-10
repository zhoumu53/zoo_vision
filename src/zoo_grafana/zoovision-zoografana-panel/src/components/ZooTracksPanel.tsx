import React, { useEffect, useRef, useState } from 'react';
import { PanelProps, DataHoverEvent } from '@grafana/data';
import { ZooTracksOptions } from 'types';
import { css, cx } from '@emotion/css';
import { useStyles2 } from '@grafana/ui';
const { DateTime } = require("luxon");

const NO_IMAGE_JPG = require('../img/no_image.jpg');

const DEFAULT_TIMESTAMP = "2025-11-15 19:08:22.308000000";
const CAMERAS = ["zag_elp_cam_017", "zag_elp_cam_018", "zag_elp_cam_016", "zag_elp_cam_019"];
const IDENTITIES = [["Indi", "Chandra"], ["Farha", "Panang"], ["Thai"]];

interface Props extends PanelProps<ZooTracksOptions> { }

interface Detection {
  timestamp: string;
  image: string;
  bbox_tlhw: number[];
  color: string;
  identity_id: number;
  identity_name: string;
  behaviour_id: number;
  behaviour_name: string;
}
type TracksState = {
  isLoading: boolean;
  detections: Detection[];
}
type CameraImageState = {
  isLoading: boolean;
  image: string;
}

const INVALID_DETECTION = {
  timestamp: "n/a",
  image: NO_IMAGE_JPG,
  bbox_tlhw: [0, 0, 0, 0],
  color: "#777777",
  identity_id: 0,
  identity_name: "Invalid",
  behaviour_id: 0,
  behaviour_name: "Invalid"
};

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
      position: relative;
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
      max-height: 128px;
    `,
    fillImage: css`
      width: 100%;
    `,
    hiddenDiv: css`
    display: none;
    `,
    grayMaskDiv: css`
      z-index: 99999;
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      width: 100%;
      opacity:0.25;
      background-color: white;
      display: flex;
      justify-content: center;
      align-items: center;
    `,
    bbox: css`
      z-index: 99998;
      position: absolute;
      border: 1px solid;
      background-color: #00000000;
    `,
    verticalCell: css`
      vertical-align: top;
      max-width: 10%;
    `,
    heatmapImg: css`
      width: 20%;
      border-style: solid;
      border-width: thin;
      border-color: gray;
    `
  };
};


export function getOffsetFromIANATimezone_min(ianaTimezone: string): number {
  const dt = DateTime.now().setZone(ianaTimezone)
  return dt.offset
}

function timestampToUtc(timestamp_s: number, timeZone: string): number {
  const utcOffset_s = getOffsetFromIANATimezone_min(timeZone) * 60;
  const timestampUtc_s = timestamp_s - utcOffset_s;
  return timestampUtc_s;
}

function buildTrackImagesUrl(track_images_server: string, cameraName: string, timestampUtc_s: number): string {
  return `${track_images_server}/track_images?camera=${cameraName}&timestamp=${timestampUtc_s}`;
}

function buildCameraImageUrl(track_images_server: string, cameraName: string, timestampUtc_s: number): string {
  return `${track_images_server}/camera_image?camera=${cameraName}&timestamp=${timestampUtc_s}`;
}

function buildWorldHeatmapUrl(track_images_server: string, start: string, end: string, identity: string): string {
  return `${track_images_server}/heatmaps/world?start_timestamp=${start}&end_timestamp=${end}&identity=${identity}`;
}

async function fetchTrackImages(url: string, abortSignal: any, setCurrentTimestamp: any, setState: (state: any) => void) {
  let new_state = {};
  let server_timestamp = "";
  try {
    const response = await fetch(url, { signal: abortSignal });
    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const dataJson = await response.json();
    if ("error" in dataJson) {
      throw new Error(dataJson["error"])
    }
    server_timestamp = dataJson["timestamp"];
    new_state = { isLoading: false, detections: dataJson["detections"] };
  } catch (err: any) {
    if (err.name === "AbortError") {
      return;
    }
    console.log("Error fetching %s: %s", url, err);
    server_timestamp = "n/a"

    new_state = { isLoading: false, detections: [INVALID_DETECTION] };
  }
  setCurrentTimestamp(server_timestamp)
  setState(new_state);
}

async function fetchCameraImages(url: string, abortSignal: any, setState: (state: any) => void) {
  let new_state = {};
  try {
    const response = await fetch(url, { signal: abortSignal });
    if (!response.ok) {
      throw new Error(response.statusText);
    }
    const dataJson = await response.json();
    if ("error" in dataJson) {
      throw new Error(dataJson["error"])
    }
    new_state = { isLoading: false, image: dataJson["image"] };
  } catch (err: any) {
    if (err.name === "AbortError") {
      return;
    }
    console.log("Error fetching %s: %s", url, err);
    new_state = { isLoading: false, image: NO_IMAGE_JPG };
  }
  setState(new_state);
}

async function changeTimestamp(track_images_server: string, controllerRef: any, timestamp_utc_s: number, setCurrentTimestamp: any, trackImagesState: TracksState[], trackImagesSetState: any[], cameraImagesState: CameraImageState[], cameraImagesSetState: any[]) {
  if (controllerRef.current) {
    controllerRef.current.abort();
  }
  controllerRef.current = new AbortController();
  const abortSignal = controllerRef.current.signal;

  for (const index in CAMERAS) {
    // Mark as loading first
    trackImagesState[index].isLoading = true;
    trackImagesSetState[index](trackImagesState[index]);
    const tracksUrl = buildTrackImagesUrl(track_images_server, CAMERAS[index], timestamp_utc_s);
    fetchTrackImages(tracksUrl, abortSignal, setCurrentTimestamp, trackImagesSetState[index]);

    // Mark as loading first
    cameraImagesState[index].isLoading = true;
    cameraImagesSetState[index](cameraImagesState[index]);
    let cameraUrl = buildCameraImageUrl(track_images_server, CAMERAS[index], timestamp_utc_s);
    fetchCameraImages(cameraUrl, abortSignal, cameraImagesSetState[index]);
  }
}

export const ZooTracksPanel: React.FC<Props> = ({ eventBus, options, width, height, replaceVariables }) => {
  const styles = useStyles2(getStyles);

  const controllerRef = useRef<AbortController>();

  // State variables
  const [currentTimestamp, setCurrentTimestamp] = useState<string>("DEFAULT");
  const emptyTrackState = { isLoading: false, detections: [] }
  const [trackImages0, trackImagesSetState0] = useState<TracksState>(emptyTrackState);
  const [trackImages1, trackImagesSetState1] = useState<TracksState>(emptyTrackState);
  const [trackImages2, trackImagesSetState2] = useState<TracksState>(emptyTrackState);
  const [trackImages3, trackImagesSetState3] = useState<TracksState>(emptyTrackState);
  const trackImages = [trackImages0, trackImages1, trackImages2, trackImages3];
  const trackImagesSetState = [trackImagesSetState0, trackImagesSetState1, trackImagesSetState2, trackImagesSetState3];

  const emptyCameraImageState = { isLoading: false, image: "" }
  const [cameraImage0, cameraImageSetState0] = useState<CameraImageState>(emptyCameraImageState);
  const [cameraImage1, cameraImageSetState1] = useState<CameraImageState>(emptyCameraImageState);
  const [cameraImage2, cameraImageSetState2] = useState<CameraImageState>(emptyCameraImageState);
  const [cameraImage3, cameraImageSetState3] = useState<CameraImageState>(emptyCameraImageState);
  const cameraImages = [cameraImage0, cameraImage1, cameraImage2, cameraImage3];
  const cameraImagesSetState = [cameraImageSetState0, cameraImageSetState1, cameraImageSetState2, cameraImageSetState3];

  if (currentTimestamp === "DEFAULT") {
    setCurrentTimestamp("");
    const time_s = timestampToUtc(Date.parse(DEFAULT_TIMESTAMP) / 1000, "Europe/Zurich");
    changeTimestamp(options.track_images_server, controllerRef, time_s, setCurrentTimestamp, trackImages, trackImagesSetState, cameraImages, cameraImagesSetState);
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
      // The event timestamp is always in browser timezone, convert to utc
      const browserTimeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
      let timestampUtc_s = timestampToUtc(timestamp_ms / 1000, browserTimeZone)
      changeTimestamp(options.track_images_server, controllerRef, timestampUtc_s, setCurrentTimestamp, trackImages, trackImagesSetState, cameraImages, cameraImagesSetState);
    });

    return () => {
      subscriber.unsubscribe();
    };
  });

  const makeTrackImages = (cameraIndex: number) => {
    return <>
      <div>Tracks</div>
      <div className={cx(styles.trackImageContainer)}>
        <div className={cx(trackImages[cameraIndex].isLoading ? styles.grayMaskDiv : styles.hiddenDiv)} >Loading</div>

        {trackImages[cameraIndex].detections.map((detection, index) =>
          <div key={index} className={cx(styles.trackImageDiv)}>
            <img src={detection.image} className={cx(styles.trackImage)} />
            <div style={{
              position: "absolute",
              backgroundColor: 'black',
              color: detection.color,
              bottom: '0%'
            }}>
              {detection.identity_name}
            </div>
          </div>
        )}
      </div>
    </>
  };
  const makeCameraImages = (cameraIndex: number) => {
    return <>
      <div>Source</div>
      <div className={cx(styles.trackImageContainer)}>
        <div className={cx(trackImages[cameraIndex].isLoading ? styles.grayMaskDiv : styles.hiddenDiv)}>Loading</div>
        {trackImages[cameraIndex].detections.map((detection, index) =>
          <div key={index} className={cx(styles.bbox)} style={{
            top: `${detection.bbox_tlhw[0] * 100}%`,
            left: `${detection.bbox_tlhw[1] * 100}% `,
            height: `${detection.bbox_tlhw[2] * 100}% `,
            width: `${detection.bbox_tlhw[3] * 100}% `,
            borderColor: `${detection.color}`,
          }}>
            <div style={{
              position: "absolute",
              backgroundColor: 'black',
              color: detection.color,
              top: '100%'
            }}>
              {detection.identity_name}
            </div>
          </div>
        )}
        <img src={cameraImages[cameraIndex].image} className={cx(styles.fillImage)} />
      </div>
    </>
  };

  return (
    <div
      className={cx(
        styles.wrapper,
        css`
          width:${width} px;
        height: ${height}px;
        overflow: auto;
        `
      )}
    >
      <div id="time-label">Time: {currentTimestamp}</div>
      <table>
        <tr>
          <th colSpan={2}><h2>Sand box mit</h2></th>
          <th colSpan={2}><h2>Sand box ohne</h2></th>
        </tr>
        <tr>
          {CAMERAS.map((cameraName) => <td key={cameraName} className={cx(styles.verticalCell)}>{cameraName}</td>)}
        </tr>
        <tr>
          {CAMERAS.map((_, index) => <td key={index} className={cx(styles.verticalCell)}>{makeTrackImages(index)}</td>)}
        </tr>
        <tr>
          {CAMERAS.map((_, index) => <td key={index} className={cx(styles.verticalCell)}>{makeCameraImages(index)}</td>)}
        </tr>
      </table>

      <h2>Occupancy heatmaps</h2>
      {IDENTITIES.map((social_group, _) => <>{
        social_group.map((identity, idx) =>
          <img key={idx} className={cx(styles.heatmapImg)} src={buildWorldHeatmapUrl(options.track_images_server, replaceVariables("${__from:date:iso}"), replaceVariables("${__to:date:iso}"), identity)} />
        )
      }
      </>)}
    </div >
  );
};

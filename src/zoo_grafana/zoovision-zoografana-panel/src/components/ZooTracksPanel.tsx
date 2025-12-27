import React, { useEffect } from 'react';
import { PanelProps, DataHoverEvent } from '@grafana/data';
import { ZooTracksOptions } from 'types';
import { css, cx } from '@emotion/css';
import { useStyles2, useTheme2 } from '@grafana/ui';
import { PanelDataErrorView } from '@grafana/runtime';

interface Props extends PanelProps<ZooTracksOptions> { }
const CAMERAS = ["zag_elp_cam_016", "zag_elp_cam_017", "zag_elp_cam_018", "zag_elp_cam_019"];

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
  };
};

export const ZooTracksPanel: React.FC<Props> = ({ eventBus, options, data, width, height, fieldConfig, id }) => {
  const styles = useStyles2(getStyles);

  useEffect(() => {
    const subscriber = eventBus.getStream(DataHoverEvent).subscribe((event) => {
      let timestamp = event.payload.point.time;
      if (timestamp == null) {
        return;
      }

      let textElement = document.getElementById('time-label');
      if (textElement != null) {
        let timestamp_s = new Date(0)
        timestamp_s.setUTCMilliseconds(timestamp)
        textElement.innerHTML = `Time: ${timestamp_s.toISOString()}`;
      }

      for (var index in CAMERAS) {
        let name = CAMERAS[index];
        let imgElement = document.getElementById(`track-image${index}`) as HTMLImageElement;
        if (imgElement != null) {
          imgElement.src = `http://127.0.0.1:5000/find_images?camera=${name}&timestamp=${timestamp}`
        }
      }
    });

    return () => {
      subscriber.unsubscribe();
    };
  }, [eventBus]);

  if (data.series.length === 0) {
    return <PanelDataErrorView fieldConfig={fieldConfig} panelId={id} data={data} needsStringField />;
  }

  let widthPerImage = width / CAMERAS.length;
  let size = (widthPerImage > height) ? height : widthPerImage;
  if (size > 512) {
    size = 512;
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
      <div id="time-label">Time:</div>
      <div className={cx(styles.rowFlex)}>
        <div className={cx(styles.areaName)}>
          <h2>
            Sand box mit
          </h2>
          <div className={cx(styles.rowFlex)}>
            <div>
              <div>Cam017</div>
              <img id="track-image1" width={size} height={size} />
            </div>
            <div>
              <div>Cam018</div>
              <img id="track-image2" width={size} height={size} />
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
              <img id="track-image0" width={size} height={size} />
            </div>
            <div>
              <div>Cam019</div>
              <img id="track-image3" width={size} height={size} />
            </div>
          </div>
        </div>
      </div>
    </div >
  );
};

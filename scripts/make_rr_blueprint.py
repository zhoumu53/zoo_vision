import rerun.blueprint as rrb


def make_blueprint():
    cameras = [
        "zag_elp_cam_016",
        "zag_elp_cam_017",
        "zag_elp_cam_018",
        "zag_elp_cam_019",
    ]

    camera_grid = rrb.Grid(
        grid_columns=2,
        contents=[
            rrb.Spatial2DView(
                origin=f"/cameras/{camera}",
                visual_bounds=rrb.VisualBounds2D(
                    # Hard-coded camera count
                    # x_range=[0, 2],
                    # y_range=[0, 1],
                ),
            )
            for camera in cameras
        ],
    )

    my_blueprint = rrb.Blueprint(
        rrb.Vertical(
            contents=[
                rrb.Horizontal(
                    contents=[
                        camera_grid,
                        rrb.Spatial2DView(
                            origin="/world",
                            visual_bounds=rrb.VisualBounds2D(
                                # Hard-coded world area
                                x_range=[-114, 11],
                                y_range=[-64, 40],
                            ),
                        ),
                    ]
                ),
                rrb.Horizontal(
                    contents=[
                        rrb.TimeSeriesView(
                            name="Frequency (Hz)",
                            origin="/processing_times/hz",
                            # axis_y=rrb.ScalarAxis(range=(0, 20), zoom_lock=False),
                        ),
                        rrb.TimeSeriesView(
                            name="Processing times (msec)",
                            origin="/processing_times/msec",
                            # axis_y=rrb.ScalarAxis(range=(0, 200), zoom_lock=False),
                        ),
                    ]
                ),
            ],
            row_shares=[0.8, 0.2],
        ),
        collapse_panels=False,
    )
    return my_blueprint


if __name__ == "__main__":
    bp = make_blueprint()
    bp.save("zoo_vision", "data/zoo_vision.rbl")

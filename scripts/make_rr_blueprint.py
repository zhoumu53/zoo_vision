import rerun.blueprint as rrb


def make_blueprint():
    cameras = [
        "zag_elp_cam_017",
        "zag_elp_cam_016",
        "zag_elp_cam_018",
        "zag_elp_cam_019",
    ]

    my_blueprint = rrb.Blueprint(
        rrb.Vertical(
            row_shares=[1, 1],
            contents=[
                rrb.Horizontal(
                    column_shares=[2, 1],
                    contents=[
                        # Camera grid
                        rrb.Grid(
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
                        ),
                        # Map view
                        rrb.Spatial2DView(
                            origin="/world",
                            visual_bounds=rrb.VisualBounds2D(
                                # Hard-coded world area
                                # x_range=[-114, 11],
                                # y_range=[-64, 40],
                                # Hard-code only sandboxes
                                x_range=[-81, -42],
                                y_range=[-71, -38],
                            ),
                        ),
                    ],
                ),
                rrb.Horizontal(
                    column_shares=[1, 12],
                    contents=[
                        rrb.TimeSeriesView(
                            name="Frequency (Hz)",
                            origin="/processing_times/hz",
                            # axis_y=rrb.ScalarAxis(range=(0, 20), zoom_lock=False),
                        ),
                        rrb.Grid(
                            grid_columns=4,
                            row_shares=[1, 2],
                            contents=[
                                rrb.TimeSeriesView(
                                    name=f"Track T{c}",
                                    origin="/tracks",
                                    contents=f"/tracks/T{c}1/votes/**",
                                )
                                for c in "abcd"
                            ]
                            + [
                                rrb.Spatial2DView(
                                    name=f"Keyframes T{c}",
                                    origin=f"/tracks/T{c}1/keyframes",
                                )
                                for c in "abcd"
                            ],
                        ),
                    ],
                ),
            ],
        ),
        collapse_panels=False,
    )
    return my_blueprint


if __name__ == "__main__":
    bp = make_blueprint()
    bp.save("zoo_vision", "data/zoo_vision.rbl")

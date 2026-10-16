"""Original source labels stay separate from audio-only inference features."""

SOURCE_CURVES = (("lossy_curves", "curve_targets", "source_curves", "original LFO curve targets", "curve"),
                 ("lossy_route_maps", "route_curve_targets", "source_route_maps", "original route maps", "route-map"),
                 ("lossy_warps", "warp_curve_targets", "source_warp_curves", "original custom warp curves", "custom-warp"))
CURVE_LABEL_KEYS = tuple(row[1] for row in SOURCE_CURVES)

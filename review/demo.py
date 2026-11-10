"""Exercise the released tile and packet code with a small RGB fixture."""

import json
from dataclasses import dataclass
from types import SimpleNamespace

from tiling import make_initial_packet, map_tile_box, tile_windows, validate_initial_packet


@dataclass(frozen=True)
class RgbFixture:
    width: int
    height: int
    pixels: bytes

    def convert(self, mode):
        if mode != "RGB":
            raise ValueError("This fixture supports RGB only.")
        return self

    def tobytes(self):
        return self.pixels


def main():
    image_size = (1440, 960)
    windows = tile_windows(image_size)
    mapped, cut_edges = map_tile_box([20, 40, 80, 100], windows[1], image_size)
    _, border_edges = map_tile_box([0, 40, 80, 100], windows[1], image_size)
    assert len(windows) == 6
    assert mapped == [500.0, 40.0, 560.0, 100.0] and not cut_edges
    assert border_edges == ["left"]

    rgb = RgbFixture(2, 2, bytes(range(12)))
    other_rgb = RgbFixture(2, 2, bytes(reversed(range(12))))
    bounds = [0, 0, 1, 1]
    box = {"xyxy": bounds, "original_xyxy": bounds, "confidence": 0.9, "class_id": 0, "class_name": "bird"}
    ordinary = SimpleNamespace(boxes=[box], passes=1, detector_candidates=1, filtered_candidates=0)
    hashes = {"example-model.pt": "0" * 64}
    settings = {"classes": [0], "conf": 0.16}
    packet = make_initial_packet(rgb, ordinary, hashes, settings, 2.0, "cpu")
    packet = json.loads(json.dumps(packet))
    accepted = validate_initial_packet(packet, rgb, hashes, settings)
    assert accepted.boxes == [box]

    try:
        validate_initial_packet(packet, other_rgb, hashes, settings)
    except ValueError:
        rejected_mismatch = True
    else:
        raise RuntimeError("Packet accepted different image pixels.")

    print(json.dumps({
        "fixture": "synthetic RGB data and one synthetic box",
        "tile_windows": len(windows),
        "original_image_box": mapped,
        "internal_edge_detected": border_edges,
        "json_handoff_boxes": len(accepted.boxes),
        "different_pixels_rejected": rejected_mismatch,
    }, indent=2))


if __name__ == "__main__":
    main()

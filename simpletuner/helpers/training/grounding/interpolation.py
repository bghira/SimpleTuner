"""Linear interpolation of bbox keyframes for video grounding."""

from __future__ import annotations


def interpolate_bbox_keyframes(
    keyframes: list[dict],
    num_frames: int,
) -> list[list[dict]]:
    """Expand keyframes into per-frame bbox_entities.

    Args:
        keyframes: List of {"frame": int, "entities": [{"label": str, "bbox": [x1,y1,x2,y2]}]}
                   Must be sorted by frame index.
        num_frames: Total number of video frames.

    Returns:
        List of length num_frames, each element is a list of
        {"label": str, "bbox": [x1,y1,x2,y2]}.
        Entities are matched across keyframes by label. Missing labels at a
        keyframe are held at their last known position.
    """
    if not keyframes or num_frames <= 0:
        return [[] for _ in range(max(num_frames, 0))]

    # Collect all unique labels and build per-label timelines
    label_timelines: dict[str, list[tuple[int, list[float]]]] = {}
    for kf in keyframes:
        frame_idx = min(kf["frame"], num_frames - 1)
        for entity in kf.get("entities", []):
            label = entity["label"]
            bbox = entity["bbox"]
            timeline = label_timelines.setdefault(label, [])
            timeline.append((frame_idx, bbox))

    # Sort each label's timeline by frame index
    for tl in label_timelines.values():
        tl.sort(key=lambda entry: entry[0])

    all_labels = sorted(label_timelines.keys())

    result: list[list[dict]] = []
    for t in range(num_frames):
        frame_entities: list[dict] = []
        for label in all_labels:
            timeline = label_timelines[label]
            bbox = _interpolate_at(timeline, t)
            frame_entities.append({"label": label, "bbox": bbox})
        result.append(frame_entities)

    return result


def _interpolate_at(timeline: list[tuple[int, list[float]]], t: int) -> list[float]:
    """Linearly interpolate bbox at frame t given a sorted timeline of (frame, bbox) pairs."""
    if len(timeline) == 1:
        return list(timeline[0][1])

    # Before first keyframe: hold first
    if t <= timeline[0][0]:
        return list(timeline[0][1])

    # After last keyframe: hold last
    if t >= timeline[-1][0]:
        return list(timeline[-1][1])

    # Find bracketing keyframes
    for i in range(len(timeline) - 1):
        f0, b0 = timeline[i]
        f1, b1 = timeline[i + 1]
        if f0 <= t <= f1:
            if f0 == f1:
                return list(b1)
            alpha = (t - f0) / (f1 - f0)
            return [b0[j] + alpha * (b1[j] - b0[j]) for j in range(4)]

    return list(timeline[-1][1])

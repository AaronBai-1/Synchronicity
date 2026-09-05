"""Canonical badminton court geometry and coordinate conventions.

Court coordinate system (pinned in docs/plan.md — do not change without a data migration):

    * origin at court centre (intersection of the net line and the centre line), metres
    * x runs across the court: x ∈ [-3.05, +3.05] (doubles sidelines)
    * y runs along the court:  y ∈ [-6.70, +6.70] (baselines)
    * the near side (closest to the broadcast camera) is y < 0
    * "left" in keypoint names means x < 0 as seen from the broadcast camera behind the near court

All homographies in this codebase map **image pixels → court metres** in this frame.
"""

from __future__ import annotations

import numpy as np

# --- BWF court dimensions (metres) -------------------------------------------------
COURT_LENGTH = 13.40
COURT_WIDTH_DOUBLES = 6.10
COURT_WIDTH_SINGLES = 5.18
HALF_LENGTH = COURT_LENGTH / 2  # 6.70

X_DOUBLES = COURT_WIDTH_DOUBLES / 2  # 3.05
X_SINGLES = COURT_WIDTH_SINGLES / 2  # 2.59
Y_BASELINE = HALF_LENGTH  # 6.70
Y_SHORT_SERVICE = 1.98  # short service line, from the net
Y_DOUBLES_LONG_SERVICE = Y_BASELINE - 0.76  # 5.94, doubles long service line

NET_HEIGHT_POSTS = 1.55
NET_HEIGHT_CENTRE = 1.524

# --- Canonical ground-plane keypoints ----------------------------------------------
# 16 line intersections on the court plane (coplanar — usable for homography fitting).
# Order is the contract for the court keypoint network's output channels.
COURT_KEYPOINTS: dict[str, tuple[float, float]] = {
    # doubles corners
    "near_corner_left": (-X_DOUBLES, -Y_BASELINE),
    "near_corner_right": (X_DOUBLES, -Y_BASELINE),
    "far_corner_left": (-X_DOUBLES, Y_BASELINE),
    "far_corner_right": (X_DOUBLES, Y_BASELINE),
    # singles sideline × baseline
    "near_singles_left": (-X_SINGLES, -Y_BASELINE),
    "near_singles_right": (X_SINGLES, -Y_BASELINE),
    "far_singles_left": (-X_SINGLES, Y_BASELINE),
    "far_singles_right": (X_SINGLES, Y_BASELINE),
    # centre line × baseline
    "near_baseline_centre": (0.0, -Y_BASELINE),
    "far_baseline_centre": (0.0, Y_BASELINE),
    # short service line × doubles sidelines
    "near_short_service_left": (-X_DOUBLES, -Y_SHORT_SERVICE),
    "near_short_service_right": (X_DOUBLES, -Y_SHORT_SERVICE),
    "far_short_service_left": (-X_DOUBLES, Y_SHORT_SERVICE),
    "far_short_service_right": (X_DOUBLES, Y_SHORT_SERVICE),
    # centre line × short service line ("T")
    "near_t_point": (0.0, -Y_SHORT_SERVICE),
    "far_t_point": (0.0, Y_SHORT_SERVICE),
}

COURT_KEYPOINT_NAMES: tuple[str, ...] = tuple(COURT_KEYPOINTS)
N_COURT_KEYPOINTS = len(COURT_KEYPOINTS)  # 16


def keypoint_array() -> np.ndarray:
    """(16, 2) float64 array of canonical keypoints in court metres, contract order."""
    return np.array([COURT_KEYPOINTS[n] for n in COURT_KEYPOINT_NAMES], dtype=np.float64)


# --- ShuttleSet-style landing area grid --------------------------------------------
# PROVISIONAL: uniform 4×4 grid per half (columns left→right in court x, rows net→baseline),
# area = row * 4 + col + 1 ∈ [1, 16]. ShuttleSet's exact area convention must be pinned
# against its homography.csv during Phase 0 golden-set work; keep this function the single
# source of truth so re-pinning is a one-place change.
_GRID_COLS = 4
_GRID_ROWS = 4


def area_from_xy(x: float, y: float) -> int | None:
    """Map a court-plane point (metres) to a 1-16 area id on its own half.

    Returns None for points outside the doubles court bounds or exactly on the net line's
    half boundary ambiguity (y == 0 counts as the far half, matching numpy floor behaviour).
    """
    if not (-X_DOUBLES <= x <= X_DOUBLES) or not (-Y_BASELINE <= abs(y) <= Y_BASELINE):
        return None
    if abs(y) > Y_BASELINE:
        return None
    col = min(int((x + X_DOUBLES) / (COURT_WIDTH_DOUBLES / _GRID_COLS)), _GRID_COLS - 1)
    row = min(int(abs(y) / (HALF_LENGTH / _GRID_ROWS)), _GRID_ROWS - 1)
    return row * _GRID_COLS + col + 1


# --- Homography utilities ----------------------------------------------------------


def fit_homography(
    image_points: np.ndarray,
    court_points: np.ndarray,
    ransac_reproj_threshold_px: float = 3.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Fit H mapping image pixels → court metres via RANSAC.

    Args:
        image_points: (N, 2) pixel coordinates of detected court keypoints, N >= 4.
        court_points: (N, 2) matching canonical court coordinates (metres).
        ransac_reproj_threshold_px: RANSAC threshold, in *pixels* (error is measured in
            the image plane by fitting the inverse mapping alongside).

    Returns:
        (H, inlier_mask) where H is (3, 3) image→court, or (None, None) if fitting failed.
    """
    import cv2

    image_points = np.asarray(image_points, dtype=np.float64)
    court_points = np.asarray(court_points, dtype=np.float64)
    if image_points.shape[0] < 4:
        return None, None
    # Fit court→image so the RANSAC threshold is in pixels, then invert.
    h_court2img, mask = cv2.findHomography(
        court_points, image_points, cv2.RANSAC, ransac_reproj_threshold_px
    )
    if h_court2img is None or abs(np.linalg.det(h_court2img)) < 1e-12:
        return None, None
    h_img2court = np.linalg.inv(h_court2img)
    h_img2court /= h_img2court[2, 2]
    return h_img2court, (mask.ravel().astype(bool) if mask is not None else None)


def project_points(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 3×3 homography to (N, 2) points."""
    points = np.asarray(points, dtype=np.float64)
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    projected = homogeneous @ np.asarray(h, dtype=np.float64).T
    return projected[:, :2] / projected[:, 2:3]


def reprojection_error_px(
    h_img2court: np.ndarray, image_points: np.ndarray, court_points: np.ndarray
) -> float:
    """RMS pixel error of the fitted homography (court points projected back to the image)."""
    h_court2img = np.linalg.inv(np.asarray(h_img2court, dtype=np.float64))
    predicted = project_points(h_court2img, court_points)
    residuals = predicted - np.asarray(image_points, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))


def homography_is_sane(
    h_img2court: np.ndarray,
    image_size: tuple[int, int],
    max_reproj_px: float = 5.0,
    image_points: np.ndarray | None = None,
    court_points: np.ndarray | None = None,
) -> bool:
    """Cheap geometric gate used for rally-camera detection (plan S1b).

    A frame is "game camera" iff court detection succeeded AND this returns True:
    the mapped court must land mostly inside the frame, right side up (near side at the
    bottom of the image), and — when correspondences are given — reproject within budget.
    """
    width, height = image_size
    h_court2img = np.linalg.inv(np.asarray(h_img2court, dtype=np.float64))
    corners_court = np.array(
        [
            COURT_KEYPOINTS["near_corner_left"],
            COURT_KEYPOINTS["near_corner_right"],
            COURT_KEYPOINTS["far_corner_left"],
            COURT_KEYPOINTS["far_corner_right"],
        ]
    )
    corners_img = project_points(h_court2img, corners_court)
    if not np.all(np.isfinite(corners_img)):
        return False
    # near baseline must appear below the far baseline (broadcast behind-court angle)
    near_y = corners_img[:2, 1].mean()
    far_y = corners_img[2:, 1].mean()
    if near_y <= far_y:
        return False
    # court must occupy a plausible share of the frame, mostly inside it
    margin = 0.35
    inside = (
        (corners_img[:, 0] > -margin * width)
        & (corners_img[:, 0] < (1 + margin) * width)
        & (corners_img[:, 1] > -margin * height)
        & (corners_img[:, 1] < (1 + margin) * height)
    )
    if inside.sum() < 3:
        return False
    quad_area = _quad_area(corners_img[[0, 1, 3, 2]])
    if not (0.03 * width * height <= quad_area <= 1.5 * width * height):
        return False
    if image_points is not None and court_points is not None:
        if reprojection_error_px(h_img2court, image_points, court_points) > max_reproj_px:
            return False
    return True


def _quad_area(quad: np.ndarray) -> float:
    """Shoelace area of a quadrilateral given as (4, 2) points in winding order."""
    x, y = quad[:, 0], quad[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

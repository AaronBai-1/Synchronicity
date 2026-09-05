import numpy as np
from synchro_pipeline.domain.court import (
    COURT_KEYPOINTS,
    N_COURT_KEYPOINTS,
    area_from_xy,
    fit_homography,
    homography_is_sane,
    keypoint_array,
    project_points,
    reprojection_error_px,
)


def test_sixteen_coplanar_keypoints_symmetric():
    assert N_COURT_KEYPOINTS == 16
    pts = dict(COURT_KEYPOINTS)
    # mirror symmetry across both axes: every keypoint's reflection is also a keypoint
    coords = set(pts.values())
    for x, y in pts.values():
        assert (-x, y) in coords
        assert (x, -y) in coords


def _synthetic_broadcast_homography() -> np.ndarray:
    """A plausible court->image projection for a 1280x720 behind-court camera."""
    court_corners = np.array([[-3.05, -6.7], [3.05, -6.7], [-3.05, 6.7], [3.05, 6.7]])
    image_corners = np.array([[140.0, 660.0], [1140.0, 660.0], [430.0, 260.0], [850.0, 260.0]])
    import cv2

    h, _ = cv2.findHomography(court_corners, image_corners)
    return h


def test_homography_round_trip():
    h_c2i = _synthetic_broadcast_homography()
    court = keypoint_array()
    image = project_points(h_c2i, court)
    h_fit, inliers = fit_homography(image, court)
    assert h_fit is not None
    assert inliers is not None and inliers.all()
    recovered = project_points(h_fit, image)
    assert np.allclose(recovered, court, atol=1e-3)  # metres; RANSAC refit is not exact
    assert reprojection_error_px(h_fit, image, court) < 0.01  # px, on perfect input


def test_fit_homography_rejects_underdetermined():
    h, mask = fit_homography(np.zeros((3, 2)), keypoint_array()[:3])
    assert h is None and mask is None


def test_homography_sanity_gate():
    h_c2i = _synthetic_broadcast_homography()
    court = keypoint_array()
    image = project_points(h_c2i, court)
    h_fit, _ = fit_homography(image, court)
    assert homography_is_sane(h_fit, (1280, 720), image_points=image, court_points=court)
    # an upside-down camera (near baseline on top) must fail the gate
    flipped = image.copy()
    flipped[:, 1] = 720 - flipped[:, 1]
    h_bad, _ = fit_homography(flipped, court)
    assert h_bad is None or not homography_is_sane(h_bad, (1280, 720))


class TestAreaGrid:
    def test_area_ids_in_range_and_cover_grid(self):
        seen = set()
        for x in np.linspace(-3.0, 3.0, 13):
            for y in np.linspace(0.05, 6.65, 13):
                area = area_from_xy(float(x), float(y))
                assert area is not None and 1 <= area <= 16
                seen.add(area)
        assert seen == set(range(1, 17))

    def test_out_of_bounds_is_none(self):
        assert area_from_xy(3.5, 2.0) is None
        assert area_from_xy(0.0, 7.0) is None

    def test_symmetric_halves(self):
        # provisional grid uses |y|: mirrored points share an area id
        assert area_from_xy(1.0, 2.0) == area_from_xy(1.0, -2.0)

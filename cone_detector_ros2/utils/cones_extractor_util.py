import numpy as np
import cupy as cp
import cv2


def get_points_only_in_bbox(boxes, points, im, classes):
    """_summary_

    Args:
        boxes (_type_): _description_
        points (_type_): _description_
        im (_type_): _description_

    Returns:
        _type_: _description_
    """
    filtered_points = []
    for box, cls in zip(boxes, classes):
        # if cls == "cone":
        #     filtered_points.append(
        #         get_points_only_in_bbox_helper_cone(box, points=points, im=im)
        #     )
        # else:
        filtered_points.append(get_points_only_in_bbox_helper_other(box, points=points))
    mask = []
    none_empty_points = []
    for i in range(len(filtered_points)):
        if cp.shape(filtered_points[i]) != () and cp.shape(filtered_points[i])[0] > 0:
            none_empty_points.append(filtered_points[i])
            mask.append(True)
        else:
            mask.append(False)
    return none_empty_points, np.array(mask)


def get_points_only_in_bbox_helper_other(bbox, points):
    """_summary_

    Args:
        bbox (_type_): _description_
        points (_type_): _description_
        im (_type_): _description_
        low (list, optional): low hsv values. Defaults to [0, 49, 166].
        high (list, optional): high hsv values. Defaults to [33, 255, 255].

    Returns:
        _type_: _description_
    """
    xmin, ymin, xmax, ymax = bbox

    mask = cp.where(
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    points_in_box = points[mask]

    result = cp.array(points_in_box)
    return result


def get_points_only_in_bbox_helper_cone(
    bbox, points, im, low=np.array([0, 49, 166]), high=np.array([33, 255, 255])
):
    """_summary_

    Args:
        bbox (_type_): _description_
        points (_type_): _description_
        im (_type_): _description_
        low (list, optional): low hsv values. Defaults to [0, 49, 166].
        high (list, optional): high hsv values. Defaults to [33, 255, 255].

    Returns:
        _type_: _description_
    """
    xmin, ymin, xmax, ymax = bbox
    # img = cp.copy(im)[:, :, :3]
    hsv_img = cv2.cvtColor(im[:, :, :3], cv2.COLOR_BGR2HSV)
    hsv_img = cp.asarray(cv2.inRange(hsv_img, low, high)) / 255

    mask = cp.where(
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    points_in_box = cp.asarray(points[mask])
    results_idx = tuple(points_in_box.astype(int)[:, :2].T.tolist())
    results = points_in_box * hsv_img[results_idx][:, None]
    results = results[~cp.all(results == 0, axis=1)]
    return results

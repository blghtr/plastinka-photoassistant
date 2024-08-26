import math
from collections import deque
from copy import copy
from typing import Dict, List, Union
import random
import cv2
import numpy as np
from numpy import ndarray

from .base_module import BaseModule
from sklearn.cluster import DBSCAN
from collections import defaultdict

class PostProcessor(BaseModule):
    def __init__(
            self,
            conf_threshold=0.5,
            max_dist=30,
            min_angle=10,
            thresh=210,
            div=30,
            div_add=5,
            div_lim=55,
            block_size=15,
            C=-2,
            gaussian_kernel_size=15,
            offset=30,
            min_line_length=100,
            max_line_gap=5,
            eps=52,
            min_samples=3,
            save_intermediate_outputs=True
    ):
        super().__init__(
            conf_threshold=conf_threshold,
            max_dist=max_dist,
            min_angle=min_angle,
            thresh=thresh,
            div=div,
            div_add=div_add,
            div_lim=div_lim,
            block_size=block_size,
            C=C,
            gaussian_kernel_size=gaussian_kernel_size,
            offset=offset,
            min_line_length=min_line_length,
            max_line_gap=max_line_gap,
            eps=eps,
            min_samples=min_samples,
            save_intermediate_outputs=save_intermediate_outputs
        )

    def _process(self, input_data: Dict) -> Dict:
        processed_border = self._fork(input_data)
        if processed_border is None:
            return None
        input_data['bbox'] = processed_border
        ofsetted_points = self._offset_box(processed_border)
        input_data['border'] = np.float32(ofsetted_points)
        return input_data

    def _fork(self, input_data: Dict) -> ndarray:
        if input_data.get('class', None) is not None:
            class_name = input_data['class'][0]
            if class_name == 'apple':
                return self._process_apple(input_data)
        return self._process_other(input_data)

    def _process_other(self, input_data: Dict) -> ndarray:
        mask = input_data['mask']
        approx = get_approximation(mask, self.args['max_dist'], self.args['min_angle'])
        if approx is None:
            return None
        sorted_points = sort_points_clockwise(approx)
        return np.float32(sorted_points)

    def _process_apple(self, input_data: Dict) -> ndarray:
        image, box = input_data['image'], input_data['box']
        box = yolo_to_box_points(box)
        angle = get_angle(
            image,
            self.args['thresh'],
            self.args['div'],
            self.args['div_add'],
            self.args['div_lim'],
            self.args['block_size'],
            self.args['C'],
            self.args['gaussian_kernel_size'],
            self.args['min_line_length'],
            self.args['max_line_gap'],
            self.args['eps'],
            self.args['min_samples']
        )
        if angle is None:
            return None
        rotated_box = rotate_box(box, angle)
        return rotated_box

    def _offset_box(self, points: ndarray) -> ndarray:
        diag1 = points[(0, 2), :]
        diag2 = points[(1, 3), :]
        center = find_intersection(diag1, diag2)

        vectors = points - center
        lengths = np.linalg.norm(vectors, axis=1)

        lengths[lengths == 0] = 1

        normalized_vectors = vectors / lengths[:, np.newaxis]
        scaled_vectors = normalized_vectors * self.args['offset']

        ofsetted_points = points + scaled_vectors
        return ofsetted_points

    def _apply_transform(self, input_data: Dict) -> ndarray:
        image = copy(input_data['image'])
        for c in (
            input_data['bbox'],
            input_data['border'],
        ):
            contours = np.array(c).reshape((-1, 1, 2)).astype(np.int32)
            image = cv2.drawContours(
                image,
                [contours],
                -1,
                [random.randint(0, 255) for _ in range(3)],
                5
            )
        return image


def calculate_angle_and_distance(p1, p2, p3):
    # This function calculates the angle at p2 formed by the line segments p1-p2 and p2-p3
    p1 = np.array(p1) if not isinstance(p1, np.ndarray) else p1
    p2 = np.array(p2) if not isinstance(p2, np.ndarray) else p2
    p3 = np.array(p3) if not isinstance(p3, np.ndarray) else p3

    a = (p1 - p2).flatten()
    b = (p3 - p2).flatten()
    angle = angle_between_vectors(a, b)
    return angle, np.linalg.norm(b)


def approximate(mask, max_dist, min_angle, find_beveled_corners=False):
    hull = cv2.convexHull(mask).astype(np.int32)
    perimeter = cv2.arcLength(hull, True)
    corrected_len = 10
    epsilon = 0.0001
    corrected_len_threshold = 4
    n_bev_c = 99
    while corrected_len > corrected_len_threshold and n_bev_c >= 4:
        epsilon *= 1.05
        scaled_eps = epsilon * perimeter
        approx = cv2.approxPolyDP(hull, scaled_eps, True).squeeze()
        approx_len = len(approx)
        beveled_corners_idx = []
        if find_beveled_corners:
            ad = {
                i: calculate_angle_and_distance(
                    approx[i - 1],
                    approx[i],
                    approx[(i + 1) % approx_len]
                ) for i in range(approx_len)
            }
            for i in range(approx_len):
                if (
                        ad[i][0] > min_angle
                        and ad[(i + 1) % approx_len][0] > min_angle
                        and ad[i][1] < max_dist
                ):
                    beveled_corners_idx.append(i)
                    n_bev_c = len(beveled_corners_idx)
            corrected_len = approx_len - len(beveled_corners_idx)
        else:
            corrected_len = approx_len

    if corrected_len < 4:
        return None, None

    return approx, beveled_corners_idx


def find_intersection(line1, line2):
    s = np.vstack([line1, line2])
    h = np.hstack((s, np.ones((4, 1))))
    l1 = np.cross(h[0], h[1])
    l2 = np.cross(h[2], h[3])
    x, y, z = np.cross(l1, l2)
    if z == 0:
        return (float('inf'), float('inf'))
    return np.asarray((x/z, y/z), dtype=np.int32)


def restore_beveled_corners(approx, beveled_corners_idx):
    beveled_corners_idx = set(beveled_corners_idx)
    approx_len = len(approx)
    new_approx = []
    for i in range(approx_len):
        if i - 1 in beveled_corners_idx:
            continue

        elif i in beveled_corners_idx:
            new_approx.append(
                find_intersection(
                    approx[(i - 1, i), :],
                    approx[((i + 1) % approx_len, (i + 2) % approx_len), :],
                )
            )
        else:
            new_approx.append(approx[i])
    return np.array(new_approx)


def get_approximation(mask, max_dist=100, min_angle=90):
    approx, beveled_corners_idx = approximate(mask, max_dist, min_angle, find_beveled_corners=True)
    if approx is None:
        return None
    if len(beveled_corners_idx):
        approx = restore_beveled_corners(approx, beveled_corners_idx)
    approx, _ = approximate(approx, max_dist, min_angle)
    return approx


def sort_points_clockwise(points: ndarray):
    # Find the start point
    sorted_by_x = np.argsort((points[:, 0]))
    y_idx = np.argmin(points[sorted_by_x][:2], 0)[1]
    start_idx = sorted_by_x[y_idx]
    start_point = copy(points[start_idx])
    rest_points = np.delete(points, start_idx, axis=0)

    # Calculate vectors from start point to all other points
    vectors = rest_points - start_point

    # Calculate angles using arctan2
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])

    # Sort points based on angles
    sorted_indices = np.argsort(angles)

    # Ensure start point is first, then add sorted points
    sorted_points = np.vstack((start_point, rest_points[sorted_indices]))

    return sorted_points


def angle_between_vectors(v1, v2):
    """
    Calculate the angle in degrees between two vectors.
    """
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    return angle_degrees

def find_lines(mask, threshold, min_line_length=100, max_line_gap=5):
    """
    Detect lines in the mask using the Hough Line Transform.
    """
    lines = cv2.HoughLinesP(mask, 1, np.pi/180, threshold=threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
    if lines is not None:
        lines = [line for line in lines if line[0, 1] != line[0, 3]]
    else:
        lines = []
    return lines


def merge_lines(lines, eps=52, min_samples=3):
    def cluster_lines(lines, eps, min_samples):
        lines_arr = np.stack(lines).reshape((-1, 4))
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(lines_arr)
        return db.labels_

    def norm(line):
        return np.linalg.norm(line[0, 2:] - line[0, :2])

    clusters = cluster_lines(lines, eps, min_samples)
    clustered_lines = [(c, l) for c, l in zip(clusters.tolist(), lines)]
    clustered_dict = defaultdict(list)
    for c, l in sorted(clustered_lines, key=lambda x: x[0]):
        if c != -1:
            clustered_dict[c].append(l)
    merged_lines = []
    for c in clustered_dict:
        old_lines = np.array(clustered_dict[c]).squeeze()
        start_x, end_x = old_lines[:, (0, 2)].min(axis=0)
        start_y, end_y = old_lines[:, (1, 3)].mean(axis=0)
        merged_lines.append(np.array(((start_x, start_y, end_x, end_y),), dtype=np.int32))
    merged_lines = sorted(merged_lines, key=norm, reverse=True)
    return merged_lines


def get_angle(
        image,
        thresh=210,
        div=30,
        div_add=5,
        div_lim=55,
        block_size=15,
        C=-2,
        gaussian_kernel_size=15,
        min_line_length=100,
        max_line_gap=5,
        eps=52,
        min_samples=3,
):
    """
    Calculate the minimum angle between the detected lines and the edges of the bounding box.
    """
    lines = []
    new_thresh = copy(thresh)
    horizontal_mask = get_horizontal_mask(image, block_size, C, div, gaussian_kernel_size)
    while not lines:
        new_thresh -= 10
        if new_thresh <= 0:
            if div + div_add >= div_lim:
                return None
            return get_angle(
                image,
                thresh=thresh,
                div=div + div_add,
                div_add=div_add,
                div_lim=div_lim,
                block_size=block_size,
                C=C,
                gaussian_kernel_size=gaussian_kernel_size,
                min_line_length=min_line_length,
                max_line_gap=max_line_gap,
                eps=eps,
                min_samples=min_samples
            )
        lines = find_lines(horizontal_mask, new_thresh, min_line_length, max_line_gap)
        if lines:
            lines = merge_lines(lines, eps=eps, min_samples=min_samples)

    longest_line = lines[0]
    line_vector = longest_line[0, 2:] - longest_line[0, :2]
    angle = np.arctan2(line_vector[1], line_vector[0]) * 180 / np.pi

    return angle

def rotate_box(box, angle):
    """
    Rotate the bounding box by the given angle.
    """
    rotation_matrix = cv2.getRotationMatrix2D(tuple(np.mean(box, axis=0)), -angle, 1)
    rotated_box = cv2.transform(np.array([box]), rotation_matrix)[0]
    return rotated_box.astype(np.int32)

def get_horizontal_mask(image, block_size=15, C=-2, div=30, gaussian_kernel_size=15):
    """
    Generate a mask highlighting horizontal lines in the image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smoothed_gray = cv2.GaussianBlur(gray, (gaussian_kernel_size, gaussian_kernel_size), 0)
    smoothed_gray = cv2.bitwise_not(smoothed_gray)
    bw = cv2.adaptiveThreshold(smoothed_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)
    horizontal = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols // div
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)
    return horizontal


def yolo_to_box_points(yolo_box):
    x1, y1, x2, y2 = yolo_box
    box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    return box


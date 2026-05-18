#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Any
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class EdgePoint:
    # 单个边缘点的最小状态：坐标、拓扑关系、梯度角度。
    x: int
    y: int
    point_id: int
    father_id: int
    angle: float
    is_root: bool = False


def dim_background(image_bgr: np.ndarray) -> np.ndarray:
    """先把背景压暗，方便边缘颜色在视觉上更突出"""
    if image_bgr.ndim == 2:
        gray = image_bgr.copy()
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = (gray.astype(np.float32) / 1.3).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def draw_single_edge(
    background: np.ndarray,
    edge: List[EdgePoint],
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 1,
) -> np.ndarray:
    """可视化单条曲线，在曲线上标注起点和终点"""
    image = background.copy()
    if len(edge) == 0:
        return image

    # 绘制曲线上的所有点
    for pt in edge:
        cv2.circle(image, (pt.x, pt.y), 1, color, thickness, cv2.LINE_AA)

    # 标注起点（绿色）
    start_pt = edge[0]
    cv2.circle(image, (start_pt.x, start_pt.y), 2, (0, 255, 0), -1)
    cv2.putText(
        image,
        "START",
        (start_pt.x + 5, start_pt.y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    # 标注终点（红色）
    if len(edge) > 1:
        end_pt = edge[-1]
        cv2.circle(image, (end_pt.x, end_pt.y), 8, (0, 0, 255), -1)
        cv2.putText(
            image,
            "END",
            (end_pt.x + 5, end_pt.y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    return image


def preprocess_canny(canny: np.ndarray) -> np.ndarray:
    """去掉"弯钩"连接点，减少后续区域生长时的错误连通"""
    cleaned = canny.copy()
    rows, cols = cleaned.shape
    for y in range(rows):
        for x in range(cols):
            if cleaned[y, x] == 0:
                continue
            left = cleaned[y, x - 1] if x > 0 else 0
            right = cleaned[y, x + 1] if x + 1 < cols else 0
            up = cleaned[y - 1, x] if y > 0 else 0
            down = cleaned[y + 1, x] if y + 1 < rows else 0
            if (
                (left > 0 and up > 0)
                or (right > 0 and up > 0)
                or (left > 0 and down > 0)
                or (right > 0 and down > 0)
            ):
                cleaned[y, x] = 0
    return cleaned


def calc_angle_bias(angle_1: float, angle_2: float) -> float:
    """计算两个角度之间的最小差值，考虑周期性（0-360度）。
    例如 10° 和 350° 的差值应该是 20°，而不是 340°。"""
    diff = abs(angle_1 - angle_2)
    return 360.0 - diff if diff > 180.0 else diff


def angle_between_lines(angle_a: float, angle_b: float) -> float:
    """
    计算两条无向直线之间的锐角（度）。

    说明：梯度角 `angle` 是有向的（0-360）。直线方向不区分正反，因此将角度模 180° 后
    取最小夹角。例如两条方向分别为 10° 和 190°（同一条直线方向相反），返回 0°。
    返回值范围为 [0, 90]。
    """
    # 把有向角转换为无向直线方向（模 180）
    a = angle_a % 180.0
    b = angle_b % 180.0
    diff = abs(a - b)
    if diff > 90.0:
        diff = 180.0 - diff
    return diff


def postProcess(
    ordered_edges: List[List[EdgePoint]],
    distance_threshold: float = 10.0,
    angle_threshold: float = 90.0,
    background: np.ndarray = None,
    waitTime: int = 10,
) -> List[List[EdgePoint]]:
    """
    从第一个曲线开始，遍历所有其他曲线：
    1. 如果当前曲线两个端点之间的距离已经小于distance_threshold，跳过它
    2. 否则找到满足距离和角度条件的其他曲线，若距离和角度差异都小于阈值则合并
    3. 合并后得到新曲线，继续寻找满足条件的曲线进行合并
    4. 若找不到满足条件的曲线，标记当前曲线，继续处理下一条曲线
    重复直到所有曲线都被处理过。

    Args:
        ordered_edges: 有序的边缘曲线列表
        distance_threshold: 端点搜索半径
        angle_threshold: 允许的最大角度差
        background: 可视化用的背景图像

    Returns:
        合并后的边缘曲线列表
    """
    if not ordered_edges:
        return []

    edges = [edge.copy() for edge in ordered_edges]
    processed_indices = set()
    merged_indices = set()

    i = 0
    while i < len(edges):
        if i in processed_indices:
            i += 1
            continue

        curve_i = edges[i]
        if len(curve_i) < 2:
            processed_indices.add(i)
            i += 1
            continue

        # ========== 可视化当前处理的曲线 ==========
        if background is not None:
            vis_image = draw_single_edge(background, curve_i, color=(0, 255, 255))
            # 在图像上标注当前处理的是哪条曲线
            h, w = vis_image.shape[:2]
            text = f"Processing Edge {i}/{len(edges)}, Points: {len(curve_i)}"
            cv2.putText(
                vis_image,
                text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Processing", vis_image)
            cv2.waitKey(waitTime)  # 等待100毫秒
        # ==========================================

        # 检查当前曲线两个端点之间的距离
        start_pt = curve_i[0]
        end_pt = curve_i[-1]
        curve_endpoints_dist = np.sqrt(
            (start_pt.x - end_pt.x) ** 2 + (start_pt.y - end_pt.y) ** 2
        )

        # 如果当前曲线两个端点之间的距离小于阈值，并且曲线长度足够，跳过它
        min_curve_length = 50
        if (
            curve_endpoints_dist < distance_threshold
            and len(curve_i) >= min_curve_length
        ):
            # 自身基本闭环
            if background is not None:
                skip_vis = draw_single_edge(background, curve_i, color=(128, 128, 128))
                cv2.putText(
                    skip_vis,
                    f"SKIP Edge {i} (dist={curve_endpoints_dist:.1f})",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("Processing", skip_vis)
                cv2.waitKey(waitTime)
            processed_indices.add(i)
            i += 1
            continue

        best_j = None
        best_curve_j = None
        best_pt_i = None
        best_pt_j = None
        min_dist = float("inf")

        for j in range(i + 1, len(edges)):
            if j in processed_indices:
                continue

            curve_j = edges[j]
            if len(curve_j) < 2:
                continue

            for pt_i in [curve_i[0], curve_i[-1]]:
                for pt_j in [curve_j[0], curve_j[-1]]:
                    dist = np.sqrt((pt_i.x - pt_j.x) ** 2 + (pt_i.y - pt_j.y) ** 2)
                    angle_diff = angle_between_lines(pt_i.angle, pt_j.angle)

                    if dist <= distance_threshold and angle_diff <= angle_threshold:
                        if dist < min_dist:
                            min_dist = dist
                            best_j = j
                            best_curve_j = curve_j
                            best_pt_i = pt_i
                            best_pt_j = pt_j

        # 如果找到满足距离和角度条件的曲线，执行合并
        if best_j is not None and min_dist <= distance_threshold:
            curve_j = best_curve_j

            # ========== 可视化即将合并的曲线 ==========
            if background is not None:
                merge_vis = draw_single_edge(background, curve_i, color=(0, 255, 255))
                merge_vis = draw_single_edge(merge_vis, curve_j, color=(255, 0, 255))
                # 标注即将合并的两条曲线
                cv2.putText(
                    merge_vis,
                    f"Edge {i} + Edge {best_j}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    merge_vis,
                    f"Distance: {min_dist:.1f} < {distance_threshold}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Processing", merge_vis)
                cv2.waitKey(waitTime)  # 等待200毫秒
            # ==========================================

            # 合并策略：根据最近端点配对决定合并方向
            # 如果最近的端点是 curve_i[0] 和 curve_j[0]（都是起点）
            # 或 curve_i[-1] 和 curve_j[-1]（都是终点）
            # 则需要反转一条曲线
            # 其他情况直接拼接
            curve_i_is_start = best_pt_i is curve_i[0]
            curve_j_is_start = best_pt_j is curve_j[0]

            if curve_i_is_start and curve_j_is_start:
                # curve_i[0] + curve_j[0] 最近，需要反转 curve_i
                merged_curve = curve_i[::-1] + curve_j
            elif not curve_i_is_start and not curve_j_is_start:
                # curve_i[-1] + curve_j[-1] 最近，需要反转 curve_j
                merged_curve = curve_i + curve_j[::-1]
            else:
                # curve_i[0] + curve_j[-1] 或 curve_i[-1] + curve_j[0]，直接拼接
                merged_curve = curve_i + curve_j

            # 重新编号
            new_curve = []
            for idx, point in enumerate(merged_curve):
                new_curve.append(
                    EdgePoint(
                        x=point.x,
                        y=point.y,
                        point_id=idx,
                        father_id=idx - 1,
                        angle=point.angle,
                        is_root=(idx == 0),
                    )
                )

            # 更新edges列表：合并后的新曲线替代当前曲线
            edges[i] = new_curve

            # 标记被合并的曲线j（不是标记i，因为i会继续使用）
            processed_indices.add(best_j)
            merged_indices.add(best_j)

            # ========== 可视化合并后的结果 ==========
            if background is not None:
                result_vis = draw_single_edge(background, new_curve, color=(0, 255, 0))
                cv2.putText(
                    result_vis,
                    f"Merged: Edge {i} ({len(new_curve)} pts)",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Processing", result_vis)
                cv2.waitKey(waitTime)
            # ==========================================

            # 合并后的新曲线作为当前曲线，继续寻找满足条件的曲线
            # 不增加i，继续以新曲线为中心
        else:
            # 没有找到满足距离和角度条件的曲线
            # 标记当前曲线为已处理
            processed_indices.add(i)
            i += 1

    if background is not None:
        cv2.destroyWindow("Processing")

    # 只移除被合并的曲线（merged_indices）
    result = [edges[i] for i in range(len(edges)) if i not in merged_indices]

    return result


def edge_tracking_8direction(
    canny, grad_angle, angle_bias=20, min_length=15
) -> Tuple[List[List[EdgePoint]], List[List[EdgePoint]]]:
    """
    8方向边缘追踪 → 双向延伸获取完整曲线
    """
    h, w = canny.shape
    visited = np.zeros_like(canny)
    edges: List[List[EdgePoint]] = []
    short_edges: List[List[EdgePoint]] = []

    # 8 方向
    dirs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if canny[y, x] == 0 or visited[y, x] == 1:
                continue

            # 1. 向一个方向生长
            path_a = []
            cx, cy = x, y
            visited[cy, cx] = 1

            while True:
                path_a.append((cx, cy))
                found = False
                for dx, dy in dirs:
                    # 注意坐标顺序：图像访问是 (y,x)，但我们习惯用 (x,y) 表示点坐标。
                    nx, ny = cx + dx, cy + dy
                    # 边界检查、必须是边缘点、未访问过
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and canny[ny, nx] == 255
                        and visited[ny, nx] == 0
                    ):
                        # 梯度方向检查：使用 calc_angle_bias 计算角度差
                        if (
                            angle_between_lines(grad_angle[cy, cx], grad_angle[ny, nx])
                            < angle_bias
                        ):
                            visited[ny, nx] = 1
                            cx, cy = nx, ny
                            found = True
                            break
                if not found:
                    break

            # 2. 向反方向生长
            path_b = []
            cx, cy = x, y
            while True:
                found = False
                for dx, dy in dirs:
                    nx, ny = cx + dx, cy + dy
                    if (
                        0 <= nx < w
                        and 0 <= ny < h
                        and canny[ny, nx] == 255
                        and visited[ny, nx] == 0
                    ):
                        # 梯度方向检查：使用 calc_angle_bias 计算角度差
                        if (
                            angle_between_lines(grad_angle[cy, cx], grad_angle[ny, nx])
                            < angle_bias
                        ):
                            visited[ny, nx] = 1
                            cx, cy = nx, ny
                            path_b.append((cx, cy))
                            found = True
                            break
                if not found:
                    break

            # 3. 组合：倒序 B + A。结果的首尾即为端点。
            full_path = path_b[::-1] + path_a
            if len(full_path) >= min_length:
                curve = []
                for i, (px, py) in enumerate(full_path):
                    curve.append(
                        EdgePoint(
                            x=px,
                            y=py,
                            point_id=i,
                            father_id=i - 1,
                            angle=float(grad_angle[py, px]),
                            is_root=(i == 0),
                        )
                    )
                edges.append(curve)
            else:
                # collect short curves for analysis/inspection
                curve = []
                for i, (px, py) in enumerate(full_path):
                    curve.append(
                        EdgePoint(
                            x=px,
                            y=py,
                            point_id=i,
                            father_id=i - 1,
                            angle=float(grad_angle[py, px]),
                            is_root=(i == 0),
                        )
                    )
                short_edges.append(curve)
    return edges, short_edges


def extract_edges(
    image_bgr: np.ndarray,
    args: Any,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[List[EdgePoint]],
    List[List[EdgePoint]],
    Dict[str, float],
]:
    """
    提取图像中的边缘主流程
    """
    # 主流程: Scharr 求梯度，Canny 找边，区域生长组织边缘。
    timing: Dict[str, float] = {}
    start_total = time.perf_counter()

    # S1: 求梯度和角度，后续区域生长时用来判断是否同一条边
    start = time.perf_counter()
    gray = (
        image_bgr
        if image_bgr.ndim == 2
        else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    )
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    _, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    timing["gradient"] = time.perf_counter() - start

    # S2: Canny 找到边缘
    start = time.perf_counter()
    canny = cv2.Canny(
        gray,
        args.canny_low,
        args.canny_high,
        apertureSize=3,
        L2gradient=True,
    )
    # 去掉弯钩连接点，减少后续区域生长时的错误连通
    if args.preprocess:
        canny = preprocess_canny(canny)
    timing["canny_preprocess"] = time.perf_counter() - start

    # S3: 区域生长边缘点，得到有序的边缘簇。
    start = time.perf_counter()
    ordered_edges, short_edges = edge_tracking_8direction(
        canny, angle, args.angle_bias, args.min_length
    )
    timing["region_grow"] = time.perf_counter() - start

    # S4: 组织有序的边缘簇，得到最终的边缘点。
    start = time.perf_counter()
    # edge_tracking_8direction 已经产出了有序边，不再需要额外的 organize()。
    # 生成用于可视化的背景图像
    background = dim_background(image_bgr)
    ordered_edges = postProcess(
        ordered_edges,
        distance_threshold=args.min_length,
        angle_threshold=args.angle_threshold,
        background=None,
    )
    timing["postProcess"] = time.perf_counter() - start
    timing["extract_total"] = time.perf_counter() - start_total
    return gray, angle, canny, ordered_edges, short_edges, timing


def build_hsv_lut() -> np.ndarray:
    """构造 HSV 色相表，用于按梯度方向着色。"""
    hsv = np.zeros((180, 1, 3), dtype=np.uint8)
    hsv[:, 0, 0] = np.arange(180, dtype=np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def build_parula_lut() -> np.ndarray:
    """用 Parula 颜色表做"沿边序号渐变"的可视化。"""
    ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
    return cv2.applyColorMap(ramp, cv2.COLORMAP_PARULA)


def draw_edges_raw(
    background: np.ndarray, canny: np.ndarray, thickness: int = 1
) -> np.ndarray:
    """使用 cv2.circle 在背景上绘制 Canny 提取的所有原始边缘点"""
    image = dim_background(background)
    coords = np.column_stack(np.where(canny > 0))
    for y, x in coords:
        cv2.circle(image, (int(x), int(y)), 1, (0, 200, 0), thickness, cv2.LINE_AA)
    return image


def draw_edges_orient(
    background: np.ndarray,
    canny: np.ndarray,
    grad_angle: np.ndarray,
    thickness: int = 1,
) -> np.ndarray:
    """用梯度角度映射到 HSV 色相，展示原始 Canny 边缘点的方向信息"""
    image = dim_background(background)
    lut = build_hsv_lut()
    coords = np.column_stack(np.where(canny > 0))
    for y, x in coords:
        angle = grad_angle[y, x]
        color = lut[int(angle / 2.0) % 180, 0].tolist()
        cv2.circle(image, (int(x), int(y)), 1, color, thickness, cv2.LINE_AA)
    return image


def draw_edges_random(
    background: np.ndarray,
    edges: Sequence[Sequence[EdgePoint]],
    thickness: int = 1,
    show_id: bool = True,
) -> np.ndarray:
    """每条边随机一个颜色，便于观察连通块之间的分割。"""
    image = dim_background(background)
    rng = np.random.default_rng(66)

    for edge_id, edge in enumerate(edges):
        color = tuple(int(c) for c in rng.integers(0, 256, size=3, dtype=np.uint8))

        # 绘制边上的所有点
        for pt in edge:
            cv2.circle(image, (pt.x, pt.y), 1, color, thickness, cv2.LINE_AA)

        # 在起始点附近添加边的ID
        if show_id and len(edge) > 0:
            start_pt = edge[0]
            # 文本位置稍微偏移，避免遮挡起点
            text_pos = (start_pt.x + 5, start_pt.y - 5)
            # 绘制黑色描边（增强可读性）
            if 1:
                cv2.putText(
                    image,
                    str(edge_id),
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                # 绘制白色文字
                cv2.putText(
                    image,
                    str(edge_id),
                    text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    return image


def draw_edges_organized(
    background: np.ndarray, edges: Sequence[Sequence[EdgePoint]], thickness: int = 1
) -> np.ndarray:
    """按边内顺序着色，颜色随点序号渐变，便于观察组织后的边序。"""
    image = dim_background(background)
    lut = build_parula_lut()
    for edge in edges:
        length = max(len(edge) - 1, 1)
        for idx, pt in enumerate(edge):
            color = lut[int(round(idx * 255 / length)), 0].tolist()
            cv2.circle(image, (pt.x, pt.y), 1, color, thickness, cv2.LINE_AA)
    return image


def make_collage(images: Sequence[np.ndarray]) -> np.ndarray:
    """合成四宫格"""
    if len(images) != 4:
        raise ValueError("expected exactly four images")

    top = np.hstack((images[0], images[1]))
    bottom = np.hstack((images[2], images[3]))
    return np.vstack((top, bottom))


def print_timing_report(
    timing: Dict[str, float], render_timing: Dict[str, float], collage_time: float
) -> None:
    """按模块输出耗时，便于快速定位慢点"""
    print("[Timing] extract_total: %.3f ms" % (timing["extract_total"] * 1000.0))
    print("[Timing]  gradient:      %.3f ms" % (timing["gradient"] * 1000.0))
    print("[Timing]  canny+preproc: %.3f ms" % (timing["canny_preprocess"] * 1000.0))
    print("[Timing]  region_grow:   %.3f ms" % (timing["region_grow"] * 1000.0))
    print("[Timing]  postProcess:   %.3f ms" % (timing["postProcess"] * 1000.0))
    print("[Timing] render_raw:     %.3f ms" % (render_timing["raw"] * 1000.0))
    print("[Timing] render_orient:  %.3f ms" % (render_timing["orient"] * 1000.0))
    print("[Timing] render_random:  %.3f ms" % (render_timing["random"] * 1000.0))
    print("[Timing] render_organ:   %.3f ms" % (render_timing["organized"] * 1000.0))
    print("[Timing] collage:        %.3f ms" % (collage_time * 1000.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize organized edges similar to examples/visEdges.cpp"
    )
    parser.add_argument(
        "--image_path", default="datasets/Real/camera_infra1_image_rect_raw/1777018836293397188.png", help="input image path"
    )
    parser.add_argument("--canny_low", type=int, default=50, help="Canny low threshold")
    parser.add_argument(
        "--canny_high", type=int, default=100, help="Canny high threshold"
    )
    parser.add_argument(
        "--min_length", type=int, default=20, help="minimum length of edges to keep"
    )
    parser.add_argument(
        "--angle_bias", type=float, default=40.0, help="angle bias in degrees"
    )
    parser.add_argument(
        "--angle_threshold",
        type=float,
        default=90.0,
        help="angle threshold for curve merging (degrees)",
    )
    parser.add_argument(
        "--thickness", type=int, default=1, help="drawing thickness for all edges"
    )
    parser.add_argument(
        "--output", type=str, default="", help="optional output image path"
    )
    parser.add_argument(
        "--preprocess",
        type=bool,
        default=False,
        help="enable canny preprocessing (hook removal)",
    )
    parser.add_argument("--no-gui", action="store_true", help="skip window display")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 打印配置参数
    print("\n" + "=" * 60)
    print("配置文件 (Configuration)")
    print("=" * 60)
    print(f"  输入图像:        {args.image_path}")
    print(f"  Canny 低阈值:    {args.canny_low}")
    print(f"  Canny 高阈值:    {args.canny_high}")
    print(f"  边缘拐角处理:    {args.preprocess}")
    print(f"  最小曲线长度:    {args.min_length}")
    print(f"  edge扩展角度差:  {args.angle_bias}°")
    print(f"  合并角度阈值:    {args.angle_threshold}°")
    print(f"  绘制线宽:        {args.thickness}")
    print(f"  输出图像:        {args.output if args.output else 'None (GUI显示)'}")
    print(f"  无GUI模式:       {args.no_gui}")
    print("=" * 60 + "\n")

    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to load image: {image_path}")

    # 打印图像尺寸信息
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    print(f"图像尺寸: {w} x {h} x {channels} (宽 x 高 x 通道数)\n")

    # 主函数：提取边缘 -> 生成四种视图 -> 拼接输出。
    _, angle, canny, edges, short_edges, timing = extract_edges(image, args)

    # 可视化
    render_timing: Dict[str, float] = {}
    start = time.perf_counter()
    raw = draw_edges_raw(image, canny, thickness=args.thickness)
    render_timing["raw"] = time.perf_counter() - start

    start = time.perf_counter()
    # 使用原始 angle 图和 canny 结果进行渲染
    orient = draw_edges_orient(image, canny, angle, thickness=args.thickness)
    render_timing["orient"] = time.perf_counter() - start

    start = time.perf_counter()
    random = draw_edges_random(image, edges, thickness=args.thickness)
    render_timing["random"] = time.perf_counter() - start

    start = time.perf_counter()
    organized = draw_edges_organized(image, edges, thickness=args.thickness)
    render_timing["organized"] = time.perf_counter() - start

    start = time.perf_counter()
    collage = make_collage(
        [
            raw,
            orient,
            random,
            organized,
        ]
    )
    collage_time = time.perf_counter() - start

    print_timing_report(timing, render_timing, collage_time)

    if args.output:
        cv2.imwrite(args.output, collage)

    if not args.no_gui:
        window_name = "O-EDGE"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        max_window_width = 1700
        max_window_height = 950
        scale = min(
            1.0,
            min(
                max_window_width / float(collage.shape[1]),
                max_window_height / float(collage.shape[0]),
            ),
        )
        cv2.resizeWindow(
            window_name, int(collage.shape[1] * scale), int(collage.shape[0] * scale)
        )
        cv2.imshow(window_name, collage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

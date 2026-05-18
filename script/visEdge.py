#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
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


class EdgeCluster:
	def __init__(self, points: Sequence[EdgePoint]):
		self.points = list(points)
		self.index_map: Dict[int, int] = {pt.point_id: idx for idx, pt in enumerate(self.points)}

	def _calculate_degree(self) -> Dict[int, int]:
		# 统计每个点的入度，用于找末端点和判断是否存在分叉。
		degree = {pt.point_id: 0 for pt in self.points}
		for pt in self.points:
			if pt.father_id >= 0 and pt.father_id in degree:
				degree[pt.father_id] += 1
		return degree

	def organize(self) -> List[EdgePoint]:
		# 按父子关系把 BFS 得到的无序点，整理成一条有序边。
		if not self.points:
			return []

		degree = self._calculate_degree()
		end_nodes = [idx for idx, pt in enumerate(self.points) if degree[pt.point_id] == 0]
		routes: List[List[EdgePoint]] = []

		# 从每个末端点回溯到根节点，形成候选路径。
		for idx in end_nodes:
			route = [self.points[idx]]
			while True:
				father_id = route[-1].father_id
				if father_id < 0:
					break
				if father_id not in self.index_map:
					break
				route.append(self.points[self.index_map[father_id]])
			routes.append(route)

		routes.sort(key=len, reverse=True)
		if len(routes) == 1:
			return routes[0]

		root_idx = next((idx for idx, pt in enumerate(self.points) if pt.is_root), None)
		if root_idx is None:
			return routes[0]

		# 根节点本身就是端点时，直接返回最长路径即可。
		if degree[self.points[root_idx].point_id] == 1:
			return routes[0]

		# 根节点出现分叉时，选取两条重叠关系最合理且总长度最大的路径进行合并。
		best_i = 0
		best_j = 0
		best_size = -1
		for i in range(len(routes)):
			for j in range(i + 1, len(routes)):
				overlap = _route_overlap(routes[i], routes[j])
				size = len(routes[i]) + len(routes[j]) - overlap
				if overlap == 1 and size > best_size:
					best_i, best_j, best_size = i, j, size

		if best_size < 0:
			return routes[0]

		merged = routes[best_j]
		merged.extend(reversed(routes[best_i][1:]))
		return merged


def _route_overlap(route_a: Sequence[EdgePoint], route_b: Sequence[EdgePoint]) -> int:
	# 逆向比较两个候选路径的父节点关系，统计尾部重叠长度。
	overlap = -1
	for pt_a, pt_b in zip(reversed(route_a), reversed(route_b)):
		if pt_a.father_id != pt_b.father_id:
			break
		overlap += 1
	return overlap


def dim_background(image_bgr: np.ndarray) -> np.ndarray:
	'''先把背景压暗，方便边缘颜色在视觉上更突出'''
	if image_bgr.ndim == 2:
		gray = image_bgr.copy()
	else:
		gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	gray = (gray.astype(np.float32) / 1.3).clip(0, 255).astype(np.uint8)
	return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def preprocess_canny(canny: np.ndarray) -> np.ndarray:
	''' 去掉“弯钩”连接点，减少后续区域生长时的错误连通 '''
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
			if (left > 0 and up > 0) or (right > 0 and up > 0) or (left > 0 and down > 0) or (right > 0 and down > 0):
				cleaned[y, x] = 0
	return cleaned


def calc_angle_bias(angle_1: float, angle_2: float) -> float:
	diff = abs(angle_1 - angle_2)
	return 360.0 - diff if diff > 180.0 else diff


def edge_tracking_8direction(canny, grad_angle, angle_bias=20):
    """
    8方向边缘追踪 → 替代 BFS
    直接输出 有序曲线 + 天然拓扑结构
    """
    h, w = canny.shape
    visited = np.zeros_like(canny)
    edges = []

    # 8 方向
    dirs = [(-1,-1), (0,-1), (1,-1),
            (-1, 0),          (1, 0),
            (-1, 1), (0, 1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if canny[y,x] == 0 or visited[y,x] == 1:
                continue

            # 找到新边缘起点
            curve = []
            cx, cy = x, y
            prev_dir = -1

            while True:
                curve.append( (cx, cy, grad_angle[cy, cx]) )
                visited[cy, cx] = 1
                found = False

                # 8方向搜索下一个点
                for d_idx, (dx, dy) in enumerate(dirs):
                    nx = cx + dx
                    ny = cy + dy
                    if 0<=nx<w and 0<=ny<h:
                        if canny[ny,nx]==255 and visited[ny,nx]==0:
                            # 角度一致性约束
                            ang1 = grad_angle[cy, cx]
                            ang2 = grad_angle[ny, nx]
                            diff = abs(ang1 - ang2)
                            diff = min(diff, 360-diff)
                            if diff < angle_bias:
                                cx, cy = nx, ny
                                prev_dir = d_idx
                                found = True
                                break
                if not found:
                    break

            if len(curve) >= 15:
                edges.append(curve)

    return edges

def region_grow(canny: np.ndarray, angle: np.ndarray, angle_bias: float) -> List[EdgeCluster]:
	'''
	以 Canny 边缘点为种子，结合梯度方向做 8 邻域 BFS 区域生长。
	这一步只负责把连通点聚成簇，真正的有序化交给 organize()。
	'''
	rows, cols = canny.shape
	visited = np.zeros((rows, cols), dtype=np.uint8)
	clusters: List[EdgeCluster] = []

	for y in range(rows):
		for x in range(cols):
			if visited[y, x] or canny[y, x] != 255:
				continue

			cluster_points: List[EdgePoint] = []
			queue = deque([(x, y, 0, -1)])
			visited[y, x] = 1
			next_point_id = 1

			while queue:
				cx, cy, point_id, father_id = queue.popleft()
				current_angle = float(angle[cy, cx])
				cluster_points.append(EdgePoint(cx, cy, point_id, father_id, current_angle, point_id == 0))

				def push(nx: int, ny: int) -> None:
					nonlocal next_point_id
					# 边界
					if nx < 0 or nx >= cols or ny < 0 or ny >= rows:
						return
					# 已访问或非边缘点
					if visited[ny, nx] or canny[ny, nx] != 255:
						return
					neigh_angle = float(angle[ny, nx])
					# 角度偏差满足要求才是同一条边，否则丢弃
					if calc_angle_bias(neigh_angle, current_angle) < angle_bias:
						visited[ny, nx] = 1
						queue.append((nx, ny, next_point_id, point_id))
						next_point_id += 1

				push(cx - 1, cy - 1)
				push(cx, cy - 1)
				push(cx + 1, cy - 1)
				push(cx - 1, cy)
				push(cx + 1, cy)
				push(cx - 1, cy + 1)
				push(cx, cy + 1)
				push(cx + 1, cy + 1)

			if len(cluster_points) > 40:
				# 小连通域通常是噪声，直接丢弃。
				clusters.append(EdgeCluster(cluster_points))

	return clusters


def extract_edges(image_bgr: np.ndarray, angle_bias: float, canny_low: int, canny_high: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[List[EdgePoint]], Dict[str, float]]:
	# 和 C++ 示例保持一致：Scharr 求梯度，Canny 找边，区域生长组织边缘。
	timing: Dict[str, float] = {}
	start_total = time.perf_counter()

	# 求梯度和角度，后续区域生长时用来判断是否同一条边
	start = time.perf_counter()
	gray = image_bgr if image_bgr.ndim == 2 else cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
	grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
	grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
	_, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
	timing["gradient"] = time.perf_counter() - start

	# canny边缘
	start = time.perf_counter()
	canny = cv2.Canny(gray, canny_low, canny_high, apertureSize=3, L2gradient=True)
	# 去掉弯钩连接点，减少后续区域生长时的错误连通
	canny = preprocess_canny(canny)
	timing["canny_preprocess"] = time.perf_counter() - start

	# 区域生长bfs边缘点，得到无序的边缘簇。
	start = time.perf_counter()
	clusters = region_grow(canny, angle, angle_bias)
	timing["region_grow"] = time.perf_counter() - start

	start = time.perf_counter()
	# 用 organize() 把交叉簇整理成有序边。
	ordered_edges = [cluster.organize() for cluster in clusters]
	timing["organize"] = time.perf_counter() - start
	timing["extract_total"] = time.perf_counter() - start_total
	return gray, angle, canny, ordered_edges, timing


def build_hsv_lut() -> np.ndarray:
	# 构造 HSV 色相表，用于按梯度方向着色。
	hsv = np.zeros((180, 1, 3), dtype=np.uint8)
	hsv[:, 0, 0] = np.arange(180, dtype=np.uint8)
	hsv[:, 0, 1] = 255
	hsv[:, 0, 2] = 255
	return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def build_parula_lut() -> np.ndarray:
	# 用 Parula 颜色表做“沿边序号渐变”的可视化。
	ramp = np.arange(256, dtype=np.uint8).reshape(-1, 1)
	return cv2.applyColorMap(ramp, cv2.COLORMAP_PARULA)


def draw_edges_raw(background: np.ndarray, edges: Sequence[Sequence[EdgePoint]]) -> np.ndarray:
	# 只画边缘点，不区分方向或顺序。
	image = dim_background(background)
	for edge in edges:
		for pt in edge:
			cv2.circle(image, (pt.x, pt.y), 1, (0, 200, 0), 1, cv2.LINE_AA)
	return image


def draw_edges_orient(background: np.ndarray, edges: Sequence[Sequence[EdgePoint]]) -> np.ndarray:
	# 用梯度角度映射到 HSV 色相，展示边缘的方向信息。
	image = dim_background(background)
	lut = build_hsv_lut()
	for edge in edges:
		for pt in edge:
			color = lut[int(pt.angle / 2.0) % 180, 0].tolist()
			cv2.circle(image, (pt.x, pt.y), 1, color, 1, cv2.LINE_AA)
	return image


def draw_edges_random(background: np.ndarray, edges: Sequence[Sequence[EdgePoint]]) -> np.ndarray:
	# 每条边随机一个颜色，便于观察连通块之间的分割。
	image = dim_background(background)
	rng = np.random.default_rng(66)
	for edge in edges:
		color = tuple(int(c) for c in rng.integers(0, 256, size=3, dtype=np.uint8))
		for pt in edge:
			cv2.circle(image, (pt.x, pt.y), 1, color, 1, cv2.LINE_AA)
	return image


def draw_edges_organized(background: np.ndarray, edges: Sequence[Sequence[EdgePoint]]) -> np.ndarray:
	# 按边内顺序着色，颜色随点序号渐变，便于观察组织后的边序。
	image = dim_background(background)
	lut = build_parula_lut()
	for edge in edges:
		length = max(len(edge) - 1, 1)
		for idx, pt in enumerate(edge):
			color = lut[int(round(idx * 255 / length)), 0].tolist()
			cv2.circle(image, (pt.x, pt.y), 1, color, 1, cv2.LINE_AA)
	return image


def make_collage(images: Sequence[np.ndarray]) -> np.ndarray:
	# 组合成四宫格，保持和 C++ 示例一致的输出形式。
	if len(images) != 4:
		raise ValueError("expected exactly four images")

	top = np.hstack((images[0], images[1]))
	bottom = np.hstack((images[2], images[3]))
	return np.vstack((top, bottom))


def print_timing_report(timing: Dict[str, float], render_timing: Dict[str, float], collage_time: float) -> None:
	# 按模块输出耗时，便于快速定位慢点。
	print("[Timing] extract_total: %.3f ms" % (timing["extract_total"] * 1000.0))
	print("[Timing]  gradient:      %.3f ms" % (timing["gradient"] * 1000.0))
	print("[Timing]  canny+preproc: %.3f ms" % (timing["canny_preprocess"] * 1000.0))
	print("[Timing]  region_grow:   %.3f ms" % (timing["region_grow"] * 1000.0))
	print("[Timing]  organize:      %.3f ms" % (timing["organize"] * 1000.0))
	print("[Timing] render_raw:     %.3f ms" % (render_timing["raw"] * 1000.0))
	print("[Timing] render_orient:  %.3f ms" % (render_timing["orient"] * 1000.0))
	print("[Timing] render_random:  %.3f ms" % (render_timing["random"] * 1000.0))
	print("[Timing] render_organ:   %.3f ms" % (render_timing["organized"] * 1000.0))
	print("[Timing] collage:        %.3f ms" % (collage_time * 1000.0))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Visualize organized edges similar to examples/visEdges.cpp")
	parser.add_argument("--image_path", default="datasets/Real/camera_infra1_image_rect_raw/1777018836226511240.png", help="input image path")
	parser.add_argument("--angle_bias", nargs="?", type=float, default=20.0, help="angle bias in degrees")
	parser.add_argument("--canny-low", type=int, default=50, help="Canny low threshold")
	parser.add_argument("--canny-high", type=int, default=100, help="Canny high threshold")
	parser.add_argument("--output", type=str, default="", help="optional output image path")
	parser.add_argument("--no-gui", action="store_true", help="skip window display")
	return parser.parse_args()


def main() -> int:
	args = parse_args()
	image_path = Path(args.image_path)
	if not image_path.is_file():
		raise FileNotFoundError(f"image not found: {image_path}")

	image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
	if image is None:
		raise RuntimeError(f"failed to load image: {image_path}")

	# 主流程：提取边缘 -> 生成四种视图 -> 拼接输出。
	_, _, _, edges, timing = extract_edges(image, args.angle_bias, args.canny_low, args.canny_high)

	render_timing: Dict[str, float] = {}
	start = time.perf_counter()
	raw = draw_edges_raw(image, edges)
	render_timing["raw"] = time.perf_counter() - start

	start = time.perf_counter()
	orient = draw_edges_orient(image, edges)
	render_timing["orient"] = time.perf_counter() - start

	start = time.perf_counter()
	random = draw_edges_random(image, edges)
	render_timing["random"] = time.perf_counter() - start

	start = time.perf_counter()
	organized = draw_edges_organized(image, edges)
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
		scale = min(1.0, min(max_window_width / float(collage.shape[1]), max_window_height / float(collage.shape[0])))
		cv2.resizeWindow(window_name, int(collage.shape[1] * scale), int(collage.shape[0] * scale))
		cv2.imshow(window_name, collage)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

import json
import base64
import math
import time
import random
from io import BytesIO
from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter, uniform_filter, sobel
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────
# Visualizations 
# ─────────────────────────────────────────────

def draw_corners_on_image(img, corners, color=(0, 255, 0), radius=4):
    pixels = img.load()
    w, h = img.size
    for cx, cy, _ in corners:
        cx, cy = int(cx), int(cy)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(math.hypot(dx, dy) - radius) < 1.0:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        pixels[px, py] = color
    return img


def draw_matches_on_images(img1, img2, kp1, kp2, matches, max_display=50):
    """Draws colored points (no lines) to indicate matching features."""
    MAX_DIM = 400
    def resize_if_needed(img):
        w, h = img.size
        if w <= MAX_DIM and h <= MAX_DIM:
            return img, 1.0
        sc = MAX_DIM / max(w, h)
        return img.resize((int(w * sc), int(h * sc)), Image.Resampling.LANCZOS), sc

    img1, s1 = resize_if_needed(img1)
    img2, s2 = resize_if_needed(img2)
    w1, h1 = img1.size
    w2, h2 = img2.size
    out_w = w1 + w2
    out_h = max(h1, h2)

    out = Image.new("RGB", (out_w, out_h), (10, 10, 12))
    out.paste(img1, (0, 0))
    out.paste(img2, (w1, 0))
    pixels = out.load()

    def draw_dot(cx, cy, r, color):
        r2 = r * r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r2:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < out_w and 0 <= py < out_h:
                        pixels[px, py] = color

    matches_sorted = sorted(matches, key=lambda m: m[2])[:max_display]
    
    random.seed(42) 
    palette = []
    for _ in range(max_display):
        palette.append((random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)))

    for rank, (i1, i2, dist) in enumerate(matches_sorted):
        if i1 >= len(kp1) or i2 >= len(kp2):
            continue
        x1_pt = int(kp1[i1][0] * s1)
        y1_pt = int(kp1[i1][1] * s1)
        x2_pt = int(kp2[i2][0] * s2) + w1
        y2_pt = int(kp2[i2][1] * s2)
        
        color = palette[rank % len(palette)]
        draw_dot(x1_pt, y1_pt, 5, color)
        draw_dot(x2_pt, y2_pt, 5, color)

    return out


# ─────────────────────────────────────────────
# Original Robust SIFT Logic
# ─────────────────────────────────────────────

def assign_orientation(cx, cy, sigma, mag, ori, width, height, num_bins=36):
    radius = int(math.ceil(3 * sigma))
    hist = [0.0] * num_bins
    bin_size = 360.0 / num_bins

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny = int(cy + dy)
            nx = int(cx + dx)
            if 0 <= nx < width and 0 <= ny < height:
                dist2 = dx * dx + dy * dy
                if dist2 <= radius * radius:
                    weight = math.exp(-dist2 / (2 * sigma * sigma))
                    bin_idx = int(ori[ny, nx] / bin_size) % num_bins
                    hist[bin_idx] += weight * mag[ny, nx]

    max_val = max(hist)
    if max_val == 0: return 0.0
    return hist.index(max_val) * bin_size


def compute_sift_descriptor(cx, cy, sigma, dominant_angle, mag, ori, width, height):
    patch_size = 16
    half_patch = patch_size // 2
    cells = 4
    cell_size = patch_size // cells
    num_bins = 8
    bin_size = 360.0 / num_bins

    descriptor = []
    sigma_weight = patch_size / 2.0

    for cell_row in range(cells):
        for cell_col in range(cells):
            hist = [0.0] * num_bins
            for py in range(cell_size):
                for px in range(cell_size):
                    scale = sigma
                    local_x = (cell_col * cell_size + px - half_patch) * scale
                    local_y = (cell_row * cell_size + py - half_patch) * scale

                    angle_rad = math.radians(-dominant_angle)
                    rot_x = local_x * math.cos(angle_rad) - local_y * math.sin(angle_rad)
                    rot_y = local_x * math.sin(angle_rad) + local_y * math.cos(angle_rad)

                    sample_x = int(round(cx + rot_x))
                    sample_y = int(round(cy + rot_y))

                    if 0 <= sample_x < width and 0 <= sample_y < height:
                        rel_ori = (ori[sample_y, sample_x] - dominant_angle) % 360.0
                        bin_idx = int(rel_ori / bin_size) % num_bins

                        dist2 = rot_x ** 2 + rot_y ** 2
                        weight = math.exp(-dist2 / (2 * sigma_weight ** 2))
                        hist[bin_idx] += weight * mag[sample_y, sample_x]

            descriptor.extend(hist)

    norm = math.sqrt(sum(v * v for v in descriptor))
    if norm > 1e-6:
        descriptor = [v / norm for v in descriptor]
    descriptor = [min(v, 0.2) for v in descriptor]
    norm = math.sqrt(sum(v * v for v in descriptor))
    if norm > 1e-6:
        descriptor = [v / norm for v in descriptor]

    return descriptor


def match_descriptors(descs1, descs2, method="ssd", ratio_threshold=0.8):
    if not descs1 or not descs2:
        return []
    
    d1 = np.array(descs1)
    d2 = np.array(descs2)
    metric = 'sqeuclidean' if method == "ssd" else 'correlation'
    dists = cdist(d1, d2, metric=metric)

    # FIX: Lowe's ratio test is based on linear Euclidean distance.
    # Since our metrics are squared (SSD and 1-NCC), we MUST square the threshold.
    effective_ratio = ratio_threshold ** 2

    matches = []
    for i, row in enumerate(dists):
        idx = np.argsort(row)
        best_j, second_best_j = idx[0], idx[1]
        best_dist, second_dist = row[best_j], row[second_best_j]

        if second_dist > 1e-9 and (best_dist / second_dist) < effective_ratio:
            if np.argmin(dists[:, best_j]) == i:
                matches.append((i, int(best_j), round(float(best_dist), 6)))
                
    return matches


# ─────────────────────────────────────────────
# OOP Services 
# ─────────────────────────────────────────────

class HarrisCornerDetector:
    def detect(self, img, method="harris", k=0.04, threshold_ratio=0.01, window_size=3, nms_size=5, sigma=1.0):
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        width, height = img.size

        t_start = time.perf_counter()
        
        gray = np.array(img.convert('L'), dtype=np.float32)
        gray = gaussian_filter(gray, sigma=sigma)
        
        Ix = sobel(gray, axis=1)
        Iy = sobel(gray, axis=0)
        
        box_area = window_size ** 2
        Ixx = uniform_filter(Ix * Ix, size=window_size) * box_area
        Iyy = uniform_filter(Iy * Iy, size=window_size) * box_area
        Ixy = uniform_filter(Ix * Iy, size=window_size) * box_area

        if method == "lambda_minus":
            a, c, b = Ixx, Iyy, Ixy
            # FIX: The mathematical discriminant for Eigenvalues is (a-c)^2 + 4b^2. 
            # The '4 *' multiplier was completely missing in the previous version!
            R = (a + c) / 2.0 - np.sqrt((a - c)**2 + 4 * b**2) / 2.0
        else:
            det = Ixx * Iyy - Ixy**2
            trace = Ixx + Iyy
            R = det - k * (trace**2)
            
        local_max = maximum_filter(R, size=nms_size) == R
        local_max = local_max & (R > 0)
        y, x = np.where(local_max)
        corners = [(int(xi), int(yi), float(R[yi, xi])) for xi, yi in zip(x, y)]

        if corners:
            max_val = max(c[2] for c in corners)
            thresh = threshold_ratio * max_val
            corners = [c for c in corners if c[2] > thresh]

        t_end = time.perf_counter()

        result_img = img.copy()
        draw_corners_on_image(result_img, corners)
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "result_image": f"data:image/png;base64,{result_b64}",
            "num_corners": len(corners),
            "computation_time_ms": round((t_end - t_start) * 1000, 2),
            "corners": corners[:200],
            "image_size": {"width": width, "height": height},
        }


class SIFTFeatureExtractor:
    def __init__(self, sigma0=1.6, num_octaves=3, num_scales=4, contrast_threshold=0.03):
        self.sigma0 = sigma0
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.contrast_threshold = contrast_threshold

    def extract(self, img, max_keypoints=150, max_dim=280):
        img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
        width, height = img.size
        gray = np.array(img.convert('L'), dtype=np.float32)

        octaves = []
        current_gray = gray
        s_factor = 2.0 ** (1.0 / self.num_scales)
        for _ in range(self.num_octaves):
            oct_scales = []
            prev = gaussian_filter(current_gray, sigma=self.sigma0)
            oct_scales.append((prev, self.sigma0))
            prev_sigma = self.sigma0
            for k in range(1, self.num_scales + 1):
                curr_sigma = self.sigma0 * (s_factor ** k)
                inc = math.sqrt(max(curr_sigma**2 - prev_sigma**2, 1e-4))
                prev = gaussian_filter(prev, sigma=inc)
                oct_scales.append((prev, curr_sigma))
                prev_sigma = curr_sigma
            octaves.append((oct_scales, current_gray.shape[1], current_gray.shape[0]))
            current_gray = current_gray[::2, ::2]

        dog_pyramid = []
        for oct_scales, ow, oh in octaves:
            dog_layers = []
            for i in range(len(oct_scales) - 1):
                dog_layers.append(oct_scales[i+1][0] - oct_scales[i][0])
            dog_pyramid.append((np.array(dog_layers), ow, oh))

        candidates = []
        for oct_idx, (dog_layers, ow, oh) in enumerate(dog_pyramid):
            if len(dog_layers) < 3: continue
            local_max = maximum_filter(dog_layers, size=3) == dog_layers
            local_min = minimum_filter(dog_layers, size=3) == dog_layers
            extrema = (local_max | local_min) & (np.abs(dog_layers) >= self.contrast_threshold)
            
            s_idx, y_idx, x_idx = np.where(extrema)
            for s, y, x in zip(s_idx, y_idx, x_idx):
                if 0 < s < len(dog_layers)-1 and 0 < y < oh-1 and 0 < x < ow-1:
                    candidates.append((abs(dog_layers[s, y, x]), oct_idx, s, x, y))

        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[:max_keypoints]

        grad_cache = {}
        for _, oct_idx, s, _, _ in candidates:
            if (oct_idx, s) not in grad_cache:
                g_img = octaves[oct_idx][0][s][0]
                Ix = np.zeros_like(g_img)
                Iy = np.zeros_like(g_img)
                Ix[:, 1:-1] = g_img[:, 2:] - g_img[:, :-2]
                Iy[1:-1, :] = g_img[2:, :] - g_img[:-2, :]
                
                mag = np.hypot(Ix, Iy)
                ori = np.degrees(np.arctan2(Iy, Ix)) % 360.0
                grad_cache[(oct_idx, s)] = (mag, ori)

        keypoints = []
        descriptors = []
        for _, oct_idx, s, x, y in candidates:
            oct_scales, ow, oh = octaves[oct_idx]
            _, sigma_oct = oct_scales[s]
            
            half_patch = 8
            border = int(math.ceil(half_patch * max(1.0, float(sigma_oct)))) + 2
            if x < border or y < border or x >= ow - border or y >= oh - border:
                continue

            mag, ori = grad_cache[(oct_idx, s)]
            desc_sigma = min(max(float(sigma_oct), 1.0), 2.5)
            dom_angle = assign_orientation(x, y, desc_sigma, mag, ori, ow, oh)
            descriptor = compute_sift_descriptor(x, y, desc_sigma, dom_angle, mag, ori, ow, oh)

            scale_factor = 2 ** oct_idx
            kx = int(x * scale_factor)
            ky = int(y * scale_factor)
            keypoints.append((kx, ky, round(float(sigma_oct * scale_factor), 3)))
            descriptors.append(descriptor)

        return {
            "image": img, "width": width, "height": height,
            "keypoints": keypoints, "descriptors": descriptors
        }


class FeatureMatcher:
    def __init__(self, extractor):
        self.extractor = extractor

    def _solve_linear_system(self, A, b):
        n = len(A)
        m = [row[:] for row in A]
        x = b[:]
        for col in range(n):
            pivot = col
            max_abs = abs(m[col][col])
            for r in range(col + 1, n):
                v = abs(m[r][col])
                if v > max_abs:
                    max_abs = v; pivot = r
            if max_abs < 1e-9: return None
            if pivot != col:
                m[col], m[pivot] = m[pivot], m[col]
                x[col], x[pivot] = x[pivot], x[col]
            pv = m[col][col]
            inv = 1.0 / pv
            for c in range(col, n): m[col][c] *= inv
            x[col] *= inv
            for r in range(n):
                if r == col: continue
                factor = m[r][col]
                if abs(factor) < 1e-12: continue
                for c in range(col, n): m[r][c] -= factor * m[col][c]
                x[r] -= factor * x[col]
        return x

    def _affine_from_3(self, p1, p2, p3, q1, q2, q3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        (u1, v1), (u2, v2), (u3, v3) = q1, q2, q3
        A = [
            [x1, y1, 1, 0, 0, 0], [0, 0, 0, x1, y1, 1],
            [x2, y2, 1, 0, 0, 0], [0, 0, 0, x2, y2, 1],
            [x3, y3, 1, 0, 0, 0], [0, 0, 0, x3, y3, 1],
        ]
        b = [u1, v1, u2, v2, u3, v3]
        sol = self._solve_linear_system(A, b)
        if sol is None: return None
        a, b_, c, d, e, f = sol
        if abs(a * e - b_ * d) < 1e-4: return None
        return a, b_, c, d, e, f

    def _apply_affine(self, params, x, y):
        a, b_, c, d, e, f = params
        return (a * x + b_ * y + c, d * x + e * y + f)

    def _filter_matches_ransac_affine(self, kp1, kp2, matches, thresh_px=8.0, iters=400, min_inliers=12):
        if len(matches) < min_inliers: return matches
        pts = [(i1, i2, d, (float(kp1[i1][0]), float(kp1[i1][1])), (float(kp2[i2][0]), float(kp2[i2][1]))) for i1, i2, d in matches]
        if len(pts) < min_inliers: return matches
        
        best_inliers = []
        thresh2 = thresh_px * thresh_px
        n = len(pts)
        iters = min(iters, n * 3)
        for t in range(iters):
            i = t % n
            j = (t * 7 + 3) % n
            k = (t * 13 + 5) % n
            if i == j or i == k or j == k: continue
            model = self._affine_from_3(pts[i][3], pts[j][3], pts[k][3], pts[i][4], pts[j][4], pts[k][4])
            if not model: continue
            inliers = []
            for ii1, ii2, dd, p, q in pts:
                ux, uy = self._apply_affine(model, p[0], p[1])
                if (q[0] - ux)**2 + (q[1] - uy)**2 <= thresh2:
                    inliers.append((ii1, ii2, dd))
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                if len(best_inliers) >= int(0.65 * n): break
                
        if len(best_inliers) >= min_inliers:
            return sorted(best_inliers, key=lambda m: m[2])
        return matches

    def _patch_ncc(self, gray_a, wa, ha, xa, ya, gray_b, wb, hb, xb, yb, radius=6):
        xa, ya, xb, yb = int(xa), int(ya), int(xb), int(yb)
        if (xa - radius < 0 or ya - radius < 0 or xa + radius >= wa or ya + radius >= ha or
            xb - radius < 0 or yb - radius < 0 or xb + radius >= wb or yb + radius >= hb):
            return -1.0
        
        patch_a = gray_a[ya-radius:ya+radius+1, xa-radius:xa+radius+1].flatten()
        patch_b = gray_b[yb-radius:yb+radius+1, xb-radius:xb+radius+1].flatten()
        
        ma, mb = np.mean(patch_a), np.mean(patch_b)
        da, db = patch_a - ma, patch_b - mb
        den = np.sqrt(np.sum(da**2) * np.sum(db**2))
        return np.sum(da * db) / den if den >= 1e-9 else -1.0

    def _filter_matches_patch_ncc(self, kp1, kp2, matches, gray1, w1, h1, gray2, w2, h2, radius=6, min_ncc=0.15):
        if len(matches) < 4: return matches
        kept = []
        for i1, i2, d in matches:
            score = self._patch_ncc(gray1, w1, h1, kp1[i1][0], kp1[i1][1], gray2, w2, h2, kp2[i2][0], kp2[i2][1], radius)
            if score >= min_ncc: kept.append((i1, i2, d))
        return kept if kept else matches

    def match(self, img1, img2, method="ssd", ratio_threshold=0.8, max_keypoints=150):
        t_sift_start = time.perf_counter()
        feat1 = self.extractor.extract(img1, max_keypoints=max_keypoints, max_dim=280)
        feat2 = self.extractor.extract(img2, max_keypoints=max_keypoints, max_dim=280)
        sift_time_ms = (time.perf_counter() - t_sift_start) * 1000

        kp1, descs1 = feat1["keypoints"], feat1["descriptors"]
        kp2, descs2 = feat2["keypoints"], feat2["descriptors"]

        t_match_start = time.perf_counter()
        matches = match_descriptors(descs1, descs2, method=method, ratio_threshold=ratio_threshold)
        
        gray1 = np.array(feat1["image"].convert('L'), dtype=np.float32)
        gray2 = np.array(feat2["image"].convert('L'), dtype=np.float32)

        if len(matches) < 40:
            matches = self._filter_matches_ransac_affine(kp1, kp2, matches, thresh_px=14.0, iters=250, min_inliers=6)
        else:
            matches = self._filter_matches_ransac_affine(kp1, kp2, matches, thresh_px=10.0, iters=350, min_inliers=10)
            
        matches = self._filter_matches_patch_ncc(kp1, kp2, matches, gray1, feat1["width"], feat1["height"], gray2, feat2["width"], feat2["height"], radius=6, min_ncc=0.15)
        match_time_ms = (time.perf_counter() - t_match_start) * 1000

        result_img = draw_matches_on_images(feat1["image"], feat2["image"], kp1, kp2, matches, max_display=50)
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        
        return {
            "result_image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}",
            "matches": matches,
            "num_matches": len(matches),
            "keypoints1": kp1, "keypoints2": kp2,
            "computation_time_ms": round(sift_time_ms + match_time_ms, 2),
            "sift_time_ms": round(sift_time_ms, 2),
            "match_time_ms": round(match_time_ms, 2),
        }


# ─────────────────────────────────────────────
# Django Views
# ─────────────────────────────────────────────

def spa(request):
    index_path = settings.BASE_DIR / "frontend" / "dist" / "index.html"
    if not index_path.is_file():
        return HttpResponse("Frontend not built. Run npm run build.", status=503)
    return FileResponse(open(index_path, "rb"), content_type="text/html; charset=utf-8")

@csrf_exempt
@require_http_methods(["POST"])
def detect_corners(request):
    data = json.loads(request.body)
    b64 = data.get("image", "").split(",", 1)[-1]
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
    
    result = HarrisCornerDetector().detect(
        img, method=data.get("method", "harris"),
        k=float(data.get("k", 0.04)),
        threshold_ratio=float(data.get("threshold_ratio", 0.01)),
        window_size=int(data.get("window_size", 3)),
        nms_size=int(data.get("nms_size", 5)),
        sigma=float(data.get("sigma", 1.0))
    )
    return JsonResponse(result)

@csrf_exempt
@require_http_methods(["POST"])
def compute_sift(request):
    data = json.loads(request.body)
    b64 = data.get("image", "").split(",", 1)[-1]
    img = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")

    extractor = SIFTFeatureExtractor(
        sigma0=float(data.get("sigma0", 1.6)),
        num_octaves=int(data.get("num_octaves", 3)),
        num_scales=int(data.get("num_scales", 4)),
        contrast_threshold=float(data.get("contrast_threshold", 0.03)),
    )
    
    t_start = time.perf_counter()
    features = extractor.extract(img, max_keypoints=int(data.get("max_keypoints", 150)))
    time_ms = round((time.perf_counter() - t_start) * 1000, 2)

    res_img = features["image"].copy()
    draw_corners_on_image(res_img, [(kp[0], kp[1], 1.0) for kp in features["keypoints"]], color=(255, 165, 0))
    buf = BytesIO()
    res_img.save(buf, format="PNG")

    return JsonResponse({
        "result_image": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}",
        "keypoints": features["keypoints"],
        "num_keypoints": len(features["keypoints"]),
        "computation_time_ms": time_ms,
        "image_size": {"width": features["width"], "height": features["height"]},
    })

@csrf_exempt
@require_http_methods(["POST"])
def match_features(request):
    data = json.loads(request.body)
    img1 = Image.open(BytesIO(base64.b64decode(data.get("image1", "").split(",", 1)[-1]))).convert("RGB")
    img2 = Image.open(BytesIO(base64.b64decode(data.get("image2", "").split(",", 1)[-1]))).convert("RGB")

    extractor = SIFTFeatureExtractor(
        sigma0=float(data.get("sigma0", 1.6)),
        num_octaves=int(data.get("num_octaves", 3)),
        num_scales=int(data.get("num_scales", 4)),
        contrast_threshold=float(data.get("contrast_threshold", 0.03)),
    )
    
    result = FeatureMatcher(extractor).match(
        img1, img2,
        method=data.get("method", "ssd"),
        ratio_threshold=float(data.get("ratio_threshold", 0.8)),
        max_keypoints=int(data.get("max_keypoints", 100))
    )
    return JsonResponse(result)
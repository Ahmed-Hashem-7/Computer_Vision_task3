import json
import base64
import math
import time
from io import BytesIO
from django.conf import settings
from django.http import FileResponse, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image


# ─────────────────────────────────────────────
# Pure-Python image helpers (no OpenCV)
# ─────────────────────────────────────────────

def image_to_grayscale(pixels, width, height):
    """Convert RGB pixel list to 2-D grayscale array."""
    gray = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[y * width + x][:3]
            gray[y][x] = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def apply_gaussian_blur(gray, width, height, sigma=1.0):
    """Apply a Gaussian blur using a 5x5 kernel."""
    radius = 2
    size = 2 * radius + 1
    kernel = [[0.0] * size for _ in range(size)]
    kernel_sum = 0.0
    for ky in range(size):
        for kx in range(size):
            dy = ky - radius
            dx = kx - radius
            val = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            kernel[ky][kx] = val
            kernel_sum += val
    for ky in range(size):
        for kx in range(size):
            kernel[ky][kx] /= kernel_sum

    blurred = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky in range(size):
                for kx in range(size):
                    ny = min(max(y + ky - radius, 0), height - 1)
                    nx = min(max(x + kx - radius, 0), width - 1)
                    acc += gray[ny][nx] * kernel[ky][kx]
            blurred[y][x] = acc
    return blurred


def apply_gaussian_blur_sigma(gray, width, height, sigma):
    """
    Generic Gaussian blur with kernel size derived from sigma.
    radius = ceil(3*sigma), ensures kernel covers ±3σ.
    """
    radius = max(1, int(math.ceil(3 * sigma)))
    size = 2 * radius + 1
    kernel = [[0.0] * size for _ in range(size)]
    kernel_sum = 0.0
    for ky in range(size):
        for kx in range(size):
            dy = ky - radius
            dx = kx - radius
            val = math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
            kernel[ky][kx] = val
            kernel_sum += val
    for ky in range(size):
        for kx in range(size):
            kernel[ky][kx] /= kernel_sum

    blurred = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            acc = 0.0
            for ky in range(size):
                for kx in range(size):
                    ny = min(max(y + ky - radius, 0), height - 1)
                    nx = min(max(x + kx - radius, 0), width - 1)
                    acc += gray[ny][nx] * kernel[ky][kx]
            blurred[y][x] = acc
    return blurred


def compute_gradients(gray, width, height):
    """Compute Ix, Iy using Sobel operators."""
    Ix = [[0.0] * width for _ in range(height)]
    Iy = [[0.0] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            Ix[y][x] = (
                -gray[y-1][x-1] + gray[y-1][x+1]
                - 2*gray[y][x-1] + 2*gray[y][x+1]
                - gray[y+1][x-1] + gray[y+1][x+1]
            )
            Iy[y][x] = (
                -gray[y-1][x-1] - 2*gray[y-1][x] - gray[y-1][x+1]
                + gray[y+1][x-1] + 2*gray[y+1][x] + gray[y+1][x+1]
            )
    return Ix, Iy


def compute_structure_tensor(Ix, Iy, width, height, window_size=3):
    """
    Compute second-moment (structure-tensor) elements Ixx, Ixy, Iyy
    by summing Ix^2, Iy^2, Ix*Iy over a square window.
    This matches the Harris M matrix from the lecture.
    """
    half = window_size // 2
    Ixx_raw = [[Ix[y][x] ** 2     for x in range(width)] for y in range(height)]
    Iyy_raw = [[Iy[y][x] ** 2     for x in range(width)] for y in range(height)]
    Ixy_raw = [[Ix[y][x]*Iy[y][x] for x in range(width)] for y in range(height)]

    Ixx = [[0.0] * width for _ in range(height)]
    Iyy = [[0.0] * width for _ in range(height)]
    Ixy = [[0.0] * width for _ in range(height)]

    for y in range(height):
        for x in range(width):
            s_xx = s_yy = s_xy = 0.0
            for wy in range(-half, half + 1):
                for wx in range(-half, half + 1):
                    ny = min(max(y + wy, 0), height - 1)
                    nx = min(max(x + wx, 0), width - 1)
                    s_xx += Ixx_raw[ny][nx]
                    s_yy += Iyy_raw[ny][nx]
                    s_xy += Ixy_raw[ny][nx]
            Ixx[y][x] = s_xx
            Iyy[y][x] = s_yy
            Ixy[y][x] = s_xy
    return Ixx, Iyy, Ixy


def compute_harris_response(Ixx, Iyy, Ixy, width, height, k=0.04):
    """
    Harris operator:
        R = det(M) - k * trace(M)^2
        det(M)   = Ixx*Iyy - Ixy^2
        trace(M) = Ixx + Iyy
    k is typically 0.04 - 0.06.
    Corner  => R >> 0 (both eigenvalues large)
    Edge    => R << 0 (one eigenvalue dominates)
    Flat    => |R| ~ 0
    """
    R = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            det   = Ixx[y][x] * Iyy[y][x] - Ixy[y][x] ** 2
            trace = Ixx[y][x] + Iyy[y][x]
            R[y][x] = det - k * trace * trace
    return R


def compute_lambda_minus(Ixx, Iyy, Ixy, width, height):
    """
    lambda- operator (minimum eigenvalue of M):
        lambda_min = (a+c)/2 - sqrt(b^2 + (a-c)^2) / 2
    where a = Ixx, c = Iyy, b = Ixy.
    From the tutorial slide on 'Fitting an Elliptical Disk'.
    Corner => lambda_min >> 0
    """
    R = [[0.0] * width for _ in range(height)]
    for y in range(height):
        for x in range(width):
            a = Ixx[y][x]
            c = Iyy[y][x]
            b = Ixy[y][x]
            R[y][x] = (a + c) / 2.0 - math.sqrt(b * b + (a - c) ** 2) / 2.0
    return R


def non_maximum_suppression(R, width, height, window_size=5):
    """Keep only local maxima within an NxN neighbourhood."""
    half = window_size // 2
    corners = []
    for y in range(half, height - half):
        for x in range(half, width - half):
            val = R[y][x]
            if val <= 0:
                continue
            is_max = True
            for wy in range(-half, half + 1):
                if not is_max:
                    break
                for wx in range(-half, half + 1):
                    if wy == 0 and wx == 0:
                        continue
                    if R[y + wy][x + wx] >= val:
                        is_max = False
                        break
            if is_max:
                corners.append((x, y, val))
    return corners


def threshold_corners(corners, threshold_ratio=0.01):
    """Keep corners whose response > threshold_ratio * max_response."""
    if not corners:
        return []
    max_val = max(c[2] for c in corners)
    threshold = threshold_ratio * max_val
    return [c for c in corners if c[2] > threshold]


def draw_corners_on_image(img, corners, color=(0, 255, 0), radius=4):
    """Draw hollow circles at corner locations on a PIL image."""
    pixels = img.load()
    w, h = img.size
    for cx, cy, _ in corners:
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                dist = math.sqrt(dx * dx + dy * dy)
                if abs(dist - radius) < 1.0:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        pixels[px, py] = color
    return img


def flatten_pixels(img):
    """Return a flat list of (r,g,b) tuples from a PIL RGB image."""
    return list(img.convert("RGB").getdata())


# ─────────────────────────────────────────────
# SIFT: Scale-Space & DoG  (from scratch)
# ─────────────────────────────────────────────

def build_scale_space(gray, width, height, sigma0=1.6, num_octaves=3, num_scales=4):
    """
    Build a Gaussian scale-space pyramid.

    For each octave we blur the image at scales:
        sigma_k = sigma0 * s^k,  k = 0 … num_scales
    where s = 2^(1/num_scales)  (constant multiplier from Tutorial 5, slide 17).

    Returns a list of octaves; each octave is a list of (blurred_2D_array, sigma).
    Between octaves the image is down-sampled by 2 in each dimension.
    """
    s = 2.0 ** (1.0 / num_scales)
    octaves = []
    current_gray = [row[:] for row in gray]
    cw, ch = width, height

    for _oct in range(num_octaves):
        oct_scales = []
        for k in range(num_scales + 1):          # +1 so DoG gives num_scales layers
            sigma_k = sigma0 * (s ** k)
            blurred = apply_gaussian_blur_sigma(current_gray, cw, ch, sigma_k)
            oct_scales.append((blurred, sigma_k))
        octaves.append((oct_scales, cw, ch))

        # Down-sample: take every other pixel for the next octave
        new_cw = cw // 2
        new_ch = ch // 2
        if new_cw < 8 or new_ch < 8:
            break
        base = oct_scales[num_scales][0]          # use the doubly-blurred layer
        downsampled = [[base[y * 2][x * 2] for x in range(new_cw)] for y in range(new_ch)]
        current_gray = downsampled
        cw, ch = new_cw, new_ch

    return octaves


def build_dog_pyramid(octaves):
    """
    Compute Difference-of-Gaussians (DoG) for each octave.
    DoG(x,y,sigma) = S(x,y,s*sigma) - S(x,y,sigma)
    This approximates (s-1)*sigma^2 * Laplacian(Gaussian) -- Tutorial 5, slide 19.

    Returns list of (dog_layers, width, height) per octave.
    Each dog_layers is a list of 2D arrays.
    """
    dog_pyramid = []
    for (oct_scales, ow, oh) in octaves:
        dog_layers = []
        for i in range(len(oct_scales) - 1):
            layer_hi = oct_scales[i + 1][0]
            layer_lo = oct_scales[i][0]
            dog = [[layer_hi[y][x] - layer_lo[y][x] for x in range(ow)] for y in range(oh)]
            dog_layers.append(dog)
        dog_pyramid.append((dog_layers, ow, oh))
    return dog_pyramid


def find_dog_extrema(dog_pyramid, contrast_threshold=0.03):
    """
    Find local extrema in the DoG pyramid across position AND scale.
    Each candidate is checked in a 3x3x3 neighbourhood (x, y, scale).
    Weak extrema below contrast_threshold are discarded.

    Returns list of (x, y, sigma, octave_index, scale_index).
    """
    keypoints = []
    for oct_idx, (dog_layers, ow, oh) in enumerate(dog_pyramid):
        num_dog = len(dog_layers)
        if num_dog < 3:
            continue
        for s in range(1, num_dog - 1):            # middle layers only
            for y in range(1, oh - 1):
                for x in range(1, ow - 1):
                    val = dog_layers[s][y][x]
                    if abs(val) < contrast_threshold:
                        continue
                    is_max = True
                    is_min = True
                    for ds in range(-1, 2):
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if ds == 0 and dy == 0 and dx == 0:
                                    continue
                                neighbour = dog_layers[s + ds][y + dy][x + dx]
                                if neighbour >= val:
                                    is_max = False
                                if neighbour <= val:
                                    is_min = False
                    if is_max or is_min:
                        # Scale sigma back to original image coordinates
                        # (multiply by 2^octave because image was down-sampled)
                        scale_factor = 2 ** oct_idx
                        orig_x = x * scale_factor
                        orig_y = y * scale_factor
                        # Approximate characteristic sigma from the octave scales list
                        sigma_approx = 1.6 * (2.0 ** (1.0 / 4)) ** s * scale_factor
                        keypoints.append((orig_x, orig_y, sigma_approx, oct_idx, s))
    return keypoints


# ─────────────────────────────────────────────
# SIFT: Orientation & Descriptor  (from scratch)
# ─────────────────────────────────────────────

def compute_gradient_magnitude_orientation(gray, width, height):
    """
    Compute per-pixel gradient magnitude and orientation (in degrees 0-360).
    Uses simple central differences.
    """
    mag = [[0.0] * width for _ in range(height)]
    ori = [[0.0] * width for _ in range(height)]
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            dx = gray[y][x + 1] - gray[y][x - 1]
            dy = gray[y + 1][x] - gray[y - 1][x]
            mag[y][x] = math.sqrt(dx * dx + dy * dy)
            ori[y][x] = math.degrees(math.atan2(dy, dx)) % 360.0
    return mag, ori


def assign_orientation(cx, cy, sigma, mag, ori, width, height, num_bins=36):
    """
    Compute the dominant gradient orientation for a keypoint.
    Uses a histogram of gradient directions within a circular window of radius 3*sigma,
    weighted by gradient magnitude.  Returns the dominant bin angle in degrees.
    (Tutorial 5, slide 23: 'Use the histogram of gradient directions'.)
    """
    radius = int(math.ceil(3 * sigma))
    hist = [0.0] * num_bins
    bin_size = 360.0 / num_bins

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny = cy + dy
            nx = cx + dx
            if 0 <= nx < width and 0 <= ny < height:
                dist2 = dx * dx + dy * dy
                if dist2 <= radius * radius:
                    weight = math.exp(-dist2 / (2 * sigma * sigma))
                    bin_idx = int(ori[ny][nx] / bin_size) % num_bins
                    hist[bin_idx] += weight * mag[ny][nx]

    max_val = max(hist)
    if max_val == 0:
        return 0.0
    dominant_bin = hist.index(max_val)
    return dominant_bin * bin_size


def compute_sift_descriptor(cx, cy, sigma, dominant_angle, mag, ori, width, height):
    """
    Build a 128-dimensional SIFT descriptor for one keypoint.

    Steps (from Tutorial 5, slides 8, 20-24):
    1. Take a 16x16 patch around the keypoint (scaled by sigma).
    2. Rotate gradient orientations relative to the dominant orientation
       (rotation invariance).
    3. Divide the patch into a 4x4 grid of 4x4 cells.
    4. For each cell, compute an 8-bin gradient-direction histogram,
       weighted by gradient magnitude and a Gaussian window.
    5. Concatenate the 4x4x8 = 128 values; L2-normalise; clip at 0.2;
       re-normalise for illumination invariance.
    """
    patch_size = 16
    half_patch = patch_size // 2
    cells = 4          # 4x4 grid
    cell_size = patch_size // cells    # 4 pixels per cell
    num_bins = 8
    bin_size = 360.0 / num_bins

    descriptor = []
    sigma_weight = patch_size / 2.0   # Gaussian weighting sigma

    for cell_row in range(cells):
        for cell_col in range(cells):
            hist = [0.0] * num_bins
            for py in range(cell_size):
                for px in range(cell_size):
                    # Compute sample coordinates in image space
                    # Scale the patch by sigma (larger keypoints => larger patch)
                    scale = sigma
                    local_x = (cell_col * cell_size + px - half_patch) * scale
                    local_y = (cell_row * cell_size + py - half_patch) * scale

                    # Rotate by dominant angle
                    angle_rad = math.radians(-dominant_angle)
                    rot_x = local_x * math.cos(angle_rad) - local_y * math.sin(angle_rad)
                    rot_y = local_x * math.sin(angle_rad) + local_y * math.cos(angle_rad)

                    sample_x = int(round(cx + rot_x))
                    sample_y = int(round(cy + rot_y))

                    if 0 <= sample_x < width and 0 <= sample_y < height:
                        # Relative orientation (rotation-invariant)
                        rel_ori = (ori[sample_y][sample_x] - dominant_angle) % 360.0
                        bin_idx = int(rel_ori / bin_size) % num_bins

                        # Gaussian weighting based on distance from keypoint
                        dist2 = rot_x ** 2 + rot_y ** 2
                        weight = math.exp(-dist2 / (2 * sigma_weight ** 2))
                        hist[bin_idx] += weight * mag[sample_y][sample_x]

            descriptor.extend(hist)

    # L2 normalise
    norm = math.sqrt(sum(v * v for v in descriptor))
    if norm > 1e-6:
        descriptor = [v / norm for v in descriptor]

    # Clip at 0.2 (illumination invariance)
    descriptor = [min(v, 0.2) for v in descriptor]

    # Re-normalise
    norm = math.sqrt(sum(v * v for v in descriptor))
    if norm > 1e-6:
        descriptor = [v / norm for v in descriptor]

    return descriptor


# ─────────────────────────────────────────────
# Feature Matching: SSD & NCC  (from scratch)
# ─────────────────────────────────────────────

def ssd_distance(desc1, desc2):
    """
    Sum of Squared Differences between two descriptors.
    Lower => better match.
    """
    return sum((a - b) ** 2 for a, b in zip(desc1, desc2))


def ncc_distance(desc1, desc2):
    """
    Normalized Cross-Correlation distance.
    We return 1 - NCC so that lower => better match (consistent with SSD).
    NCC = (desc1 · desc2) / (||desc1|| * ||desc2||)
    Since SIFT descriptors are already L2-normalised, NCC = dot product.
    """
    dot = sum(a * b for a, b in zip(desc1, desc2))
    # Clamp to [-1, 1] to handle floating-point drift
    dot = max(-1.0, min(1.0, dot))
    return 1.0 - dot


def match_descriptors(descs1, descs2, method="ssd", ratio_threshold=0.8):
    """
    Match two sets of descriptors using Lowe's ratio test.

    For each descriptor in descs1 we find the two nearest neighbours in descs2
    (by SSD or NCC distance). A match is accepted only when:
        best_distance / second_best_distance < ratio_threshold

    Returns list of (idx1, idx2, distance) for accepted matches.
    """
    dist_fn = ssd_distance if method == "ssd" else ncc_distance

    # Forward: for each descriptor in descs1, find best/second-best in descs2
    forward_best = []
    for i, d1 in enumerate(descs1):
        best_dist   = float("inf")
        second_dist = float("inf")
        best_j      = -1
        for j, d2 in enumerate(descs2):
            dist = dist_fn(d1, d2)
            if dist < best_dist:
                second_dist = best_dist
                best_dist   = dist
                best_j      = j
            elif dist < second_dist:
                second_dist = dist
        forward_best.append((best_j, best_dist, second_dist))

    # Backward: for each descriptor in descs2, find best in descs1
    backward_best = []
    for j, d2 in enumerate(descs2):
        best_dist = float("inf")
        best_i    = -1
        for i, d1 in enumerate(descs1):
            dist = dist_fn(d1, d2)
            if dist < best_dist:
                best_dist = dist
                best_i    = i
        backward_best.append((best_i, best_dist))

    # Keep only mutual-best matches that also satisfy Lowe's ratio test
    matches = []
    for i, (best_j, best_dist, second_dist) in enumerate(forward_best):
        if best_j == -1 or second_dist <= 1e-9:
            continue
        ratio = best_dist / second_dist
        if ratio >= ratio_threshold:
            continue
        back_i, _ = backward_best[best_j]
        if back_i != i:
            continue
        matches.append((i, best_j, round(best_dist, 6)))

    return matches


def draw_matches_on_images(img1, img2, kp1, kp2, matches, max_display=50):
    """
    Create a side-by-side image with matched keypoints connected by lines.
    kp1 / kp2: list of (x, y, sigma, …)
    matches: list of (idx1, idx2, dist)
    Returns a PIL Image.
    """
    # Downscale very large inputs so drawing stays fast and legible
    MAX_DIM = 400

    def resize_if_needed(img):
        w, h = img.size
        if w <= MAX_DIM and h <= MAX_DIM:
            return img, 1.0
        s = MAX_DIM / max(w, h)
        return img.resize((int(w * s), int(h * s)), Image.LANCZOS), s

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

    def draw_circle(cx, cy, r, color):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(math.sqrt(dx * dx + dy * dy) - r) < 1.0:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < out_w and 0 <= py < out_h:
                        pixels[px, py] = color

    # Sort matches by distance (best first) and keep at most max_display
    matches_sorted = sorted(matches, key=lambda m: m[2])[:max_display]

    palette = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
        (255, 160, 0), (160, 0, 255),
    ]

    for rank, (i1, i2, dist) in enumerate(matches_sorted):
        if i1 >= len(kp1) or i2 >= len(kp2):
            continue
        # Rescale coordinates if images were resized
        x1_pt = int(kp1[i1][0] * s1)
        y1_pt = int(kp1[i1][1] * s1)
        x2_pt = int(kp2[i2][0] * s2) + w1
        y2_pt = int(kp2[i2][1] * s2)

        color = palette[rank % len(palette)]
        # Draw filled circles only (no connecting line) so the image stays clean
        draw_circle(x1_pt, y1_pt, 4, color)
        draw_circle(x2_pt, y2_pt, 4, color)

    return out


# ─────────────────────────────────────────────
# OOP services
# ─────────────────────────────────────────────

class HarrisCornerDetector:
    def detect(self, img, method="harris", k=0.04, threshold_ratio=0.01, window_size=3, nms_size=5, sigma=1.0):
        width, height = img.size
        max_dim = 400
        if width > max_dim or height > max_dim:
            scale = max_dim / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            img = img.resize((width, height), Image.LANCZOS)

        t_start = time.perf_counter()
        flat = flatten_pixels(img)
        gray = image_to_grayscale(flat, width, height)
        gray = apply_gaussian_blur(gray, width, height, sigma=sigma)
        ix, iy = compute_gradients(gray, width, height)
        ixx, iyy, ixy = compute_structure_tensor(ix, iy, width, height, window_size=window_size)
        if method == "lambda_minus":
            response = compute_lambda_minus(ixx, iyy, ixy, width, height)
        else:
            response = compute_harris_response(ixx, iyy, ixy, width, height, k=k)
        corners = non_maximum_suppression(response, width, height, window_size=nms_size)
        corners = threshold_corners(corners, threshold_ratio=threshold_ratio)
        t_end = time.perf_counter()

        result_img = img.copy()
        draw_corners_on_image(result_img, corners, color=(0, 255, 0), radius=4)
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "result_image": f"data:image/png;base64,{result_b64}",
            "num_corners": len(corners),
            "computation_time_ms": round((t_end - t_start) * 1000, 2),
            "corners": [[c[0], c[1], round(c[2], 4)] for c in corners[:200]],
            "image_size": {"width": width, "height": height},
        }


class SIFTFeatureExtractor:
    def __init__(self, sigma0=1.6, num_octaves=3, num_scales=4, contrast_threshold=0.03):
        self.sigma0 = sigma0
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.contrast_threshold = contrast_threshold

    def _resize(self, img, max_dim):
        w, h = img.size
        if w > max_dim or h > max_dim:
            s = max_dim / max(w, h)
            img = img.resize((int(w * s), int(h * s)), Image.LANCZOS)
        return img

    def extract(self, img, max_keypoints=200, max_dim=350):
        img = self._resize(img, max_dim=max_dim)
        width, height = img.size
        gray = image_to_grayscale(flatten_pixels(img), width, height)
        octaves = build_scale_space(gray, width, height, self.sigma0, self.num_octaves, self.num_scales)
        dog_pyramid = build_dog_pyramid(octaves)

        candidates = []
        for oct_idx, (dog_layers, ow, oh) in enumerate(dog_pyramid):
            num_dog = len(dog_layers)
            if num_dog < 3:
                continue
            for s in range(1, num_dog - 1):
                for y in range(1, oh - 1):
                    for x in range(1, ow - 1):
                        val = dog_layers[s][y][x]
                        if abs(val) < self.contrast_threshold:
                            continue
                        is_max = True
                        is_min = True
                        for ds in range(-1, 2):
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    if ds == 0 and dy == 0 and dx == 0:
                                        continue
                                    nval = dog_layers[s + ds][y + dy][x + dx]
                                    if nval >= val:
                                        is_max = False
                                    if nval <= val:
                                        is_min = False
                        if is_max or is_min:
                            candidates.append((abs(val), oct_idx, s, x, y))

        candidates.sort(key=lambda t: t[0], reverse=True)
        candidates = candidates[:max_keypoints]

        grad_cache = {}
        for oct_idx, (oct_scales, ow, oh) in enumerate(octaves):
            for s_idx, (g_img, _sigma_oct) in enumerate(oct_scales):
                grad_cache[(oct_idx, s_idx)] = compute_gradient_magnitude_orientation(g_img, ow, oh)

        keypoints = []
        descriptors = []
        for _strength, oct_idx, s, x, y in candidates:
            oct_scales, ow, oh = octaves[oct_idx]
            _g_img, sigma_oct = oct_scales[s]
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
            "image": img,
            "width": width,
            "height": height,
            "keypoints": keypoints,
            "descriptors": descriptors,
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
                    max_abs = v
                    pivot = r
            if max_abs < 1e-9:
                return None
            if pivot != col:
                m[col], m[pivot] = m[pivot], m[col]
                x[col], x[pivot] = x[pivot], x[col]
            pv = m[col][col]
            inv = 1.0 / pv
            for c in range(col, n):
                m[col][c] *= inv
            x[col] *= inv
            for r in range(n):
                if r == col:
                    continue
                factor = m[r][col]
                if abs(factor) < 1e-12:
                    continue
                for c in range(col, n):
                    m[r][c] -= factor * m[col][c]
                x[r] -= factor * x[col]
        return x

    def _affine_from_3(self, p1, p2, p3, q1, q2, q3):
        (x1, y1), (x2, y2), (x3, y3) = p1, p2, p3
        (u1, v1), (u2, v2), (u3, v3) = q1, q2, q3
        A = [
            [x1, y1, 1, 0, 0, 0],
            [0, 0, 0, x1, y1, 1],
            [x2, y2, 1, 0, 0, 0],
            [0, 0, 0, x2, y2, 1],
            [x3, y3, 1, 0, 0, 0],
            [0, 0, 0, x3, y3, 1],
        ]
        b = [u1, v1, u2, v2, u3, v3]
        sol = self._solve_linear_system(A, b)
        if sol is None:
            return None
        a, b_, c, d, e, f = sol
        det = a * e - b_ * d
        if abs(det) < 1e-4:
            return None
        return a, b_, c, d, e, f

    @staticmethod
    def _apply_affine(params, x, y):
        a, b_, c, d, e, f = params
        return (a * x + b_ * y + c, d * x + e * y + f)

    def _filter_matches_ransac_affine(self, kp1, kp2, matches, thresh_px=8.0, iters=400, min_inliers=12):
        if len(matches) < min_inliers:
            return matches
        pts = []
        for i1, i2, d in matches:
            if i1 >= len(kp1) or i2 >= len(kp2):
                continue
            pts.append((i1, i2, d, (float(kp1[i1][0]), float(kp1[i1][1])), (float(kp2[i2][0]), float(kp2[i2][1]))))
        if len(pts) < min_inliers:
            return matches
        best_inliers = []
        thresh2 = thresh_px * thresh_px
        n = len(pts)
        iters = min(iters, n * 3)
        for t in range(iters):
            i = t % n
            j = (t * 7 + 3) % n
            k = (t * 13 + 5) % n
            if i == j or i == k or j == k:
                continue
            p1 = pts[i][3]; p2 = pts[j][3]; p3 = pts[k][3]
            q1 = pts[i][4]; q2 = pts[j][4]; q3 = pts[k][4]
            model = self._affine_from_3(p1, p2, p3, q1, q2, q3)
            if model is None:
                continue
            inliers = []
            for ii1, ii2, dd, p, q in pts:
                ux, uy = self._apply_affine(model, p[0], p[1])
                dx = q[0] - ux
                dy = q[1] - uy
                if dx * dx + dy * dy <= thresh2:
                    inliers.append((ii1, ii2, dd))
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                if len(best_inliers) >= int(0.65 * n):
                    break
        if len(best_inliers) >= min_inliers:
            return sorted(best_inliers, key=lambda m: m[2])
        return matches

    @staticmethod
    def _patch_ncc(gray_a, wa, ha, xa, ya, gray_b, wb, hb, xb, yb, radius=6):
        xa = int(xa); ya = int(ya); xb = int(xb); yb = int(yb)
        if (
            xa - radius < 0 or ya - radius < 0 or xa + radius >= wa or ya + radius >= ha or
            xb - radius < 0 or yb - radius < 0 or xb + radius >= wb or yb + radius >= hb
        ):
            return -1.0
        a_vals = []
        b_vals = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                a_vals.append(gray_a[ya + dy][xa + dx])
                b_vals.append(gray_b[yb + dy][xb + dx])
        n = len(a_vals)
        ma = sum(a_vals) / n
        mb = sum(b_vals) / n
        num = 0.0
        da2 = 0.0
        db2 = 0.0
        for i in range(n):
            da = a_vals[i] - ma
            db = b_vals[i] - mb
            num += da * db
            da2 += da * da
            db2 += db * db
        den = math.sqrt(da2 * db2)
        if den < 1e-9:
            return -1.0
        return num / den

    def _filter_matches_patch_ncc(self, kp1, kp2, matches, gray1, w1, h1, gray2, w2, h2, radius=6, min_ncc=0.15):
        if len(matches) < 4:
            return matches
        kept = []
        for i1, i2, d in matches:
            if i1 >= len(kp1) or i2 >= len(kp2):
                continue
            x1, y1 = kp1[i1][0], kp1[i1][1]
            x2, y2 = kp2[i2][0], kp2[i2][1]
            score = self._patch_ncc(gray1, w1, h1, x1, y1, gray2, w2, h2, x2, y2, radius=radius)
            if score >= min_ncc:
                kept.append((i1, i2, d))
        return kept if kept else matches

    def match(self, img1, img2, method="ssd", ratio_threshold=0.8, max_keypoints=150):
        t_sift_start = time.perf_counter()
        feat1 = self.extractor.extract(img1, max_keypoints=max_keypoints, max_dim=380)
        feat2 = self.extractor.extract(img2, max_keypoints=max_keypoints, max_dim=380)
        t_sift_end = time.perf_counter()
        sift_time_ms = (t_sift_end - t_sift_start) * 1000

        kp1, descs1 = feat1["keypoints"], feat1["descriptors"]
        kp2, descs2 = feat2["keypoints"], feat2["descriptors"]
        img1r, img2r = feat1["image"], feat2["image"]
        w1, h1 = feat1["width"], feat1["height"]
        w2, h2 = feat2["width"], feat2["height"]
        gray1 = image_to_grayscale(flatten_pixels(img1r), w1, h1)
        gray2 = image_to_grayscale(flatten_pixels(img2r), w2, h2)

        t_match_start = time.perf_counter()
        if descs1 and descs2:
            matches = match_descriptors(descs1, descs2, method=method, ratio_threshold=ratio_threshold)
        else:
            matches = []
        if len(matches) < 40:
            matches = self._filter_matches_ransac_affine(kp1, kp2, matches, thresh_px=14.0, iters=700, min_inliers=8)
        else:
            matches = self._filter_matches_ransac_affine(kp1, kp2, matches, thresh_px=10.0, iters=600, min_inliers=12)
        matches = self._filter_matches_patch_ncc(kp1, kp2, matches, gray1, w1, h1, gray2, w2, h2, radius=6, min_ncc=0.15)
        t_match_end = time.perf_counter()
        match_time_ms = (t_match_end - t_match_start) * 1000

        result_img = draw_matches_on_images(img1r, img2r, kp1, kp2, matches, max_display=50)
        buf = BytesIO()
        result_img.save(buf, format="PNG")
        result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return {
            "result_image": f"data:image/png;base64,{result_b64}",
            "matches": matches,
            "num_matches": len(matches),
            "keypoints1": [[kp[0], kp[1], kp[2]] for kp in kp1],
            "keypoints2": [[kp[0], kp[1], kp[2]] for kp in kp2],
            "computation_time_ms": round(sift_time_ms + match_time_ms, 2),
            "sift_time_ms": round(sift_time_ms, 2),
            "match_time_ms": round(match_time_ms, 2),
        }


# ─────────────────────────────────────────────
# Django views
# ─────────────────────────────────────────────

def spa(request):
    """Serve the Vite-built React app (run `npm run build` in `frontend/`)."""
    index_path = settings.BASE_DIR / "frontend" / "dist" / "index.html"
    if not index_path.is_file():
        return HttpResponse(
            "Frontend not built. From the project root run:\n\n"
            "  cd frontend\n"
            "  npm install\n"
            "  npm run build\n\n"
            "Then reload this page.",
            status=503,
            content_type="text/plain; charset=utf-8",
        )
    return FileResponse(open(index_path, "rb"), content_type="text/html; charset=utf-8")


@csrf_exempt
@require_http_methods(["POST"])
def detect_corners(request):
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    b64 = data.get("image", "")
    if not b64:
        return JsonResponse({"error": "No image provided."}, status=400)

    method          = data.get("method", "harris")
    k               = float(data.get("k", 0.04))
    threshold_ratio = float(data.get("threshold_ratio", 0.01))
    window_size     = max(3, int(data.get("window_size", 3)))
    nms_size        = max(3, int(data.get("nms_size", 5)))
    sigma           = float(data.get("sigma", 1.0))

    # Decode the base64 image
    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return JsonResponse({"error": f"Could not decode image: {e}"}, status=400)
    detector = HarrisCornerDetector()
    result = detector.detect(
        img,
        method=method,
        k=k,
        threshold_ratio=threshold_ratio,
        window_size=window_size,
        nms_size=nms_size,
        sigma=sigma,
    )
    return JsonResponse(result)


@csrf_exempt
@require_http_methods(["POST"])
def compute_sift(request):
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    b64 = data.get("image", "")
    if not b64:
        return JsonResponse({"error": "No image provided."}, status=400)

    sigma0              = float(data.get("sigma0", 1.6))
    num_octaves         = int(data.get("num_octaves", 3))
    num_scales          = int(data.get("num_scales", 4))
    contrast_threshold  = float(data.get("contrast_threshold", 0.03))
    max_keypoints       = int(data.get("max_keypoints", 200))

    try:
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return JsonResponse({"error": f"Could not decode image: {e}"}, status=400)
    extractor = SIFTFeatureExtractor(
        sigma0=sigma0,
        num_octaves=num_octaves,
        num_scales=num_scales,
        contrast_threshold=contrast_threshold,
    )
    t_start = time.perf_counter()
    features = extractor.extract(img, max_keypoints=max_keypoints, max_dim=350)
    t_end = time.perf_counter()

    result_img = features["image"].copy()
    draw_corners_on_image(result_img, [(kp[0], kp[1], 1.0) for kp in features["keypoints"]], color=(255, 165, 0), radius=5)
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    result_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JsonResponse({
        "result_image": f"data:image/png;base64,{result_b64}",
        "keypoints": [[kp[0], kp[1], kp[2]] for kp in features["keypoints"]],
        "descriptors": [[round(v, 5) for v in desc] for desc in features["descriptors"]],
        "num_keypoints": len(features["keypoints"]),
        "computation_time_ms": round((t_end - t_start) * 1000, 2),
        "image_size": {"width": features["width"], "height": features["height"]},
    })


@csrf_exempt
@require_http_methods(["POST"])
def match_features(request):
    try:
        data = json.loads(request.body)
    except Exception:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)

    b64_1 = data.get("image1", "")
    b64_2 = data.get("image2", "")
    if not b64_1 or not b64_2:
        return JsonResponse({"error": "Both image1 and image2 are required."}, status=400)

    method             = data.get("method", "ssd")
    ratio_threshold    = float(data.get("ratio_threshold", 0.8))
    sigma0             = float(data.get("sigma0", 1.6))
    num_octaves        = int(data.get("num_octaves", 3))
    num_scales         = int(data.get("num_scales", 4))
    contrast_threshold = float(data.get("contrast_threshold", 0.03))
    max_keypoints      = int(data.get("max_keypoints", 150))

    def decode_image(b64):
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        img_bytes = base64.b64decode(b64)
        return Image.open(BytesIO(img_bytes)).convert("RGB")

    try:
        img1 = decode_image(b64_1)
        img2 = decode_image(b64_2)
    except Exception as e:
        return JsonResponse({"error": f"Could not decode image: {e}"}, status=400)
    extractor = SIFTFeatureExtractor(
        sigma0=sigma0,
        num_octaves=num_octaves,
        num_scales=num_scales,
        contrast_threshold=contrast_threshold,
    )
    matcher = FeatureMatcher(extractor)
    result = matcher.match(
        img1,
        img2,
        method=method,
        ratio_threshold=ratio_threshold,
        max_keypoints=max_keypoints,
    )
    return JsonResponse(result)
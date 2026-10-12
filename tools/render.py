"""Tiny numpy software rasteriser used for previews and depot icons.

Draws textured triangles with a z-buffer, simple sun + sky lighting and a gloss highlight.
"""
import math
import numpy as np


def look_at(eye, target, up=(0, 0, 1)):
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, r, u, f


class Renderer:
    def __init__(self, w, h, eye, target, fov=30.0, ortho=None, ss=2, bg=None):
        self.W, self.H, self.ss = w * ss, h * ss, ss
        self.out_size = (w, h)
        self.eye, self.r, self.u, self.f = look_at(eye, target)
        self.fov = fov
        self.ortho = ortho  # half height in metres for orthographic projection
        self.z = np.full((self.H, self.W), np.inf)
        self.col = np.zeros((self.H, self.W, 3))
        self.mask = np.zeros((self.H, self.W), bool)
        if bg is not None:
            self.col[:] = bg
        # key light from behind the camera, slightly left and above
        self.sun = -self.f * 0.55 - self.r * 0.35 + np.array([0, 0, 0.75])
        self.sun /= np.linalg.norm(self.sun)

    def project(self, P):
        d = P - self.eye
        x = d @ self.r
        y = d @ self.u
        z = d @ self.f
        if self.ortho:
            s = self.H / 2 / self.ortho
            return np.stack([self.W / 2 + x * s, self.H / 2 - y * s, z], -1)
        s = self.H / 2 / math.tan(math.radians(self.fov) / 2)
        zz = np.maximum(z, 1e-3)
        return np.stack([self.W / 2 + x / zz * s, self.H / 2 - y / zz * s, z], -1)

    def draw(self, P, UV, N, tex, mga=None, alpha=1.0, backface=True):
        """P (T,3,3), UV (T,3,2), N (T,3,3); tex (H,W,3 floats 0..1) or rgb tuple."""
        if len(P) == 0:
            return
        S = self.project(P.reshape(-1, 3)).reshape(-1, 3, 3)
        if backface:
            e1 = S[:, 1, :2] - S[:, 0, :2]
            e2 = S[:, 2, :2] - S[:, 0, :2]
            cr = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
            keep = cr < 0  # screen y points down -> CCW becomes negative
            S, P, UV, N = S[keep], P[keep], UV[keep], N[keep]
        if not self.ortho:
            ok = (S[:, :, 2] > 0.1).all(1)
            S, P, UV, N = S[ok], P[ok], UV[ok], N[ok]
        istex = not isinstance(tex, tuple)
        for t in range(len(S)):
            s = S[t]
            x0 = int(max(0, math.floor(s[:, 0].min())))
            x1 = int(min(self.W - 1, math.ceil(s[:, 0].max())))
            y0 = int(max(0, math.floor(s[:, 1].min())))
            y1 = int(min(self.H - 1, math.ceil(s[:, 1].max())))
            if x1 < x0 or y1 < y0:
                continue
            yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            px = xx + 0.5
            py = yy + 0.5
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = s
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            w0 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / den
            w1 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / den
            w2 = 1 - w0 - w1
            inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
            if not inside.any():
                continue
            if self.ortho:
                b0, b1, b2 = w0, w1, w2
            else:
                q0, q1, q2 = w0 / az, w1 / bz, w2 / cz
                qs = q0 + q1 + q2
                b0, b1, b2 = q0 / qs, q1 / qs, q2 / qs
            depth = b0 * az + b1 * bz + b2 * cz
            zb = self.z[y0:y1 + 1, x0:x1 + 1]
            vis = inside & (depth < zb)
            if not vis.any():
                continue
            bb0, bb1, bb2 = b0[vis][:, None], b1[vis][:, None], b2[vis][:, None]
            n = bb0 * N[t, 0] + bb1 * N[t, 1] + bb2 * N[t, 2]
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
            if istex:
                uv = bb0 * UV[t, 0] + bb1 * UV[t, 1] + bb2 * UV[t, 2]
                th, tw = tex.shape[:2]
                iu = np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)
                iv = np.clip((uv[:, 1] * th).astype(int), 0, th - 1)
                c = tex[iv, iu]
                gl = mga[iv, iu, 1] if mga is not None else np.full(len(c), 0.4)
            else:
                c = np.broadcast_to(np.array(tex, float), (len(n), 3))
                gl = np.full(len(c), 0.4)
            pos = bb0 * P[t, 0] + bb1 * P[t, 1] + bb2 * P[t, 2]
            view = self.eye - pos
            view /= np.linalg.norm(view, axis=1, keepdims=True) + 1e-9
            if not backface:
                flip = (n * view).sum(1) < 0
                n[flip] *= -1
            diff = np.clip(n @ self.sun, 0, 1)
            sky = 0.42 + 0.22 * n[:, 2]
            h = self.sun + view
            h /= np.linalg.norm(h, axis=1, keepdims=True)
            spec = np.clip((n * h).sum(1), 0, 1) ** (8 + 120 * gl) * gl * 0.9
            shade = c * (0.85 * diff[:, None] + sky[:, None]) + spec[:, None]
            region = self.col[y0:y1 + 1, x0:x1 + 1]
            if alpha < 1:
                region[vis] = region[vis] * (1 - alpha) + shade * alpha
            else:
                region[vis] = shade
                zb[vis] = depth[vis]
                self.mask[y0:y1 + 1, x0:x1 + 1][vis] = True

    def image(self):
        img = np.clip(self.col, 0, 1)
        ss = self.ss
        H, W = self.out_size[1], self.out_size[0]
        img = img.reshape(H, ss, W, ss, 3).mean((1, 3))
        a = self.mask.reshape(H, ss, W, ss).mean((1, 3))
        return (img ** (1 / 1.15) * 255).astype(np.uint8), (a * 255).astype(np.uint8)

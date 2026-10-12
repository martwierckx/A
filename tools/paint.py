"""Procedural textures: body livery baked from the 3D surface, swatches, interior, glass."""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import railjet as rj
from swatch import Swatches

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

C = {
    'red_dark': (178, 16, 30),
    'red': (205, 20, 32),
    'red_bright': (232, 28, 38),
    'grey_low': (72, 72, 75),
    'grey_light': (140, 140, 142),
    'roof': (42, 42, 45),
    'black': (22, 22, 24),
    'under': (34, 34, 36),
    'frame': (210, 212, 214),
    'door': (62, 62, 66),
    'glass': (70, 96, 118),
    'white': (235, 235, 235),
}


def body_swatches(atlas_px):
    sw = Swatches(atlas_px, (0.0, 0.875, 1.0, 1.0), 64)
    sw.define('wheel_steel', (95, 92, 88), metal=0.8, gloss=0.45)
    sw.define('wheel_tread', (150, 148, 145), metal=1.0, gloss=0.7)
    sw.define('bogie_dark', (38, 38, 40), metal=0.2, gloss=0.3)
    sw.define('bogie_frame', (48, 48, 52), metal=0.1, gloss=0.35)
    sw.define('rubber', (24, 24, 25), gloss=0.2)
    sw.define('axlebox_cover', (230, 180, 20), gloss=0.5)
    sw.define('spring', (60, 60, 62), metal=0.3, gloss=0.3,
              pattern=lambda t, u, v: t * (0.55 + 0.45 * (np.sin(v * 40)[..., None] > 0)))
    sw.define('insulator', (120, 60, 40), gloss=0.7,
              pattern=lambda t, u, v: t * (0.7 + 0.3 * (np.sin(v * 50)[..., None] > 0)))
    sw.define('panto_blue', (44, 70, 120), metal=0.3, gloss=0.5)
    sw.define('panto_alu', (180, 182, 186), metal=0.9, gloss=0.6)
    sw.define('panto_carbon', (30, 30, 30), gloss=0.2)
    sw.define('bellows', (30, 30, 34), gloss=0.15,
              pattern=lambda t, u, v: t * (0.75 + 0.25 * (np.sin(u * 30)[..., None] > 0)))
    sw.define('bellows_frame', (24, 24, 26), gloss=0.4)
    sw.define('coupler_box', (190, 190, 192), gloss=0.4)
    sw.define('roof_box', (40, 40, 43), gloss=0.3)
    sw.define('grille', (55, 55, 58), metal=0.4, gloss=0.4,
              pattern=lambda t, u, v: t * (0.5 + 0.5 * ((np.sin(u * 90) > 0) ^ (np.sin(v * 90) > 0))[..., None]))
    sw.define('gasket', (18, 18, 18), gloss=0.5)
    sw.define('door_in', (150, 150, 150), gloss=0.3)
    sw.define('endwall', C['black'], gloss=0.3)
    return sw


def interior_swatches(atlas_px):
    sw = Swatches(atlas_px, (0.0, 0.0, 1.0, 1.0), 64)

    def fabric(base, accent):
        def f(t, u, v):
            chk = ((np.floor(u * 8) + np.floor(v * 8)) % 2)[..., None]
            noise = (np.sin(u * 140) * np.sin(v * 160))[..., None] * 0.06
            return t * (0.9 + noise) * (1 - chk * 0.12) + np.array(accent) * chk * 0.12
        return f
    sw.define('wall', (216, 216, 212), gloss=0.35)
    sw.define('floor', (60, 62, 68), gloss=0.15,
              pattern=lambda t, u, v: t * (0.9 + 0.1 * (np.sin(u * 60) * np.sin(v * 60) > 0)[..., None]))
    sw.define('ceiling', (238, 238, 236), gloss=0.3)
    sw.define('partition', (200, 210, 215), metal=0.2, gloss=0.8)
    sw.define('step', (80, 82, 88), gloss=0.2,
              pattern=lambda t, u, v: np.where((v % 0.25 < 0.04)[..., None], np.array([200, 160, 30.0]), t))
    sw.define('seat2', (52, 56, 66), gloss=0.15, pattern=fabric((52, 56, 66), (120, 20, 30)))
    sw.define('seat2_head', (176, 20, 32), gloss=0.2)
    sw.define('seat1', (74, 62, 58), gloss=0.35, pattern=fabric((74, 62, 58), (40, 34, 30)))
    sw.define('seat1_head', (40, 36, 34), gloss=0.4)
    sw.define('seat_frame', (90, 92, 96), metal=0.6, gloss=0.5)
    sw.define('table', (150, 120, 90), gloss=0.5)
    sw.define('rack', (170, 172, 176), metal=0.7, gloss=0.6)
    sw.define('bike', (40, 40, 40), metal=0.4, gloss=0.3)
    sw.define('cab_floor', (50, 50, 52), gloss=0.2)
    sw.define('cab_wall', (78, 80, 86), gloss=0.3)
    sw.define('desk', (45, 46, 50), gloss=0.4)
    sw.define('screen', (40, 80, 120), gloss=0.9)
    sw.define('seat_cab', (30, 30, 34), gloss=0.3)
    return sw


# ---------------------------------------------------------------------------- body bake
def _text_stamp(text, h_px, color, bold=True):
    font = ImageFont.truetype(FONT, h_px)
    l, t, r, b = font.getbbox(text)
    img = Image.new('RGBA', (r - l + 4, b - t + 4), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - l, 2 - t), text, font=font, fill=color + (255,))
    return np.asarray(img).astype(float)


def _paste(rgb, stamp, cx, cy, mirror=False):
    if mirror:
        stamp = stamp[:, ::-1]
    h, w = stamp.shape[:2]
    x0 = int(cx - w / 2)
    y0 = int(cy - h / 2)
    H, W = rgb.shape[:2]
    xa, ya = max(0, x0), max(0, y0)
    xb, yb = min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    s = stamp[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[..., 3:4] / 255
    rgb[ya:yb, xa:xb] = rgb[ya:yb, xa:xb] * (1 - a) + s[..., :3] * a


def _roundrect_d(px, pz, x0, x1, z0, z1, r):
    """Signed distance (metres) to a rounded rectangle; negative inside."""
    cx, cz = (x0 + x1) / 2, (z0 + z1) / 2
    hx, hz = (x1 - x0) / 2 - r, (z1 - z0) / 2 - r
    dx = np.abs(px - cx) - hx
    dz = np.abs(pz - cz) - hz
    out = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dz, 0) ** 2)
    ins = np.minimum(np.maximum(dx, dz), 0)
    return out + ins - r


def bake_loft_chart(spec, chart, atlas, mga, height, dirt, px, cls, lights):
    """Paint one loft chart (rows chart[0]..chart[1] of the atlas)."""
    r0, r1 = int(round(chart[0] * px)), int(round(chart[1] * px))
    xs = -13.6 + (np.arange(px) + 0.5) / px * 27.2
    vv = ((np.arange(r0, r1) + 0.5) / px - chart[0]) / (chart[1] - chart[0])
    jf = np.interp(vv, rj.VNOM, np.arange(rj.J))
    S = spec.section3d(xs)                                  # (X, J, 3)
    j0 = np.clip(np.floor(jf).astype(int), 0, rj.J - 2)
    fr = (jf - j0)[:, None]
    X3 = S[:, j0, 0].T * (1 - fr) + S[:, j0 + 1, 0].T * fr  # (R, X)
    Y = S[:, j0, 1].T * (1 - fr) + S[:, j0 + 1, 1].T * fr
    Z = S[:, j0, 2].T * (1 - fr) + S[:, j0 + 1, 2].T * fr
    X = np.broadcast_to(xs[None, :], Z.shape)                # nominal station (texture u)
    J2 = np.broadcast_to(jf[:, None], Z.shape)
    R, W = Z.shape
    col = np.zeros((R, W, 3))
    gl = np.full((R, W), 0.62)
    met = np.zeros((R, W))
    ao = np.ones((R, W))
    hgt = np.zeros((R, W))

    def put(mask, c, g=None, m=None):
        col[mask] = c
        if g is not None:
            gl[mask] = g
        if m is not None:
            met[mask] = m

    end = spec.kind == 'end'
    # base livery: dark red top, bright stripe, grey bottom
    put(np.ones_like(Z, bool), C['red_dark'])
    put(Z < 2.14, C['red_bright'])
    put(Z < 1.92, C['grey_low'], 0.55)
    if end:
        tn = np.clip((X - rj.XS) / (rj.XT - rj.XS), 0, 1)
        # swoosh: the bright stripe turns down in front of the front door to the body bottom;
        # dark red in front of it, grey behind it with a lighter grey wedge
        zz = np.clip((Z - 0.30) / 1.65, 0, 1)
        xsw = 6.55 - 1.45 * zz ** 1.3
        below = Z < 2.14
        put(below & (X3 >= xsw), C['red_bright'])
        put(below & (X3 >= xsw + 0.50 + 0.25 * (1 - zz)), C['red_dark'])
        wedge = (Z < 1.92) & (X3 >= xsw - 0.55) & (X3 < xsw - 0.15) & (Z > 0.55)
        put(wedge, C['grey_light'], 0.6)
        # black front mask: windscreen frame incl. the shelf below it
        put((tn >= 0.49) & (J2 >= 8.6) & (J2 <= 13.7), C['black'], 0.85)
        # black parallelogram around the cab side window (follows the windscreen rake)
        put((tn >= 0.06) & (tn <= 0.29) & (J2 >= 9.6) & (J2 <= 12.6), C['black'], 0.85)
        # skirt and coupler recess
        put((X3 > 9.4) & (Z < 0.82), C['under'], 0.3)
        # small air intake grilles beside the coupler recess and panel shut lines of the chin
        chin_front = (X3 > 13.0) & (Z > 1.0) & (Z < 2.1)
        for k in range(3):
            zz0 = 1.88 + 0.06 * k
            put(chin_front & (np.abs(np.abs(Y) - 0.78) < 0.12) & (np.abs(Z - zz0) < 0.015), C['black'], 0.4)
        seam = (X3 > 12.0) & (np.abs(Z - 0.97) < 0.006)
        col[seam] *= 0.55
        hgt[seam] = -1
        for (a, b, cells) in spec.recess:
            put((X >= a) & (np.floor(J2) == cells[0]), C['black'], 0.3)
        # cab door
        cd = _roundrect_d(X3, Z, 8.95, 9.60, 1.30, 3.20, 0.08)
        side = Y > 1.2
        put((np.abs(cd) < 0.012) & side, C['black'])
        hgt[(np.abs(cd) < 0.012) & side] = -1
        put((_roundrect_d(X3, Z, 9.08, 9.36, 2.20, 2.95, 0.06) < 0) & side, C['glass'], 0.95)
        # ventilation grille high up on the side behind the cab
        g = (X3 > 7.45) & (X3 < 8.35) & (Z > 3.45) & (Z < 3.85) & side
        put(g, C['black'], 0.4)
        put(g & ((Z * 40).astype(int) % 2 == 0), (60, 60, 64), 0.4)
        # headlight bezels (3D distance, flat ellipse)
        for (p, r) in lights:
            dy = np.abs(Y) - abs(p[1])
            dz = (Z - p[2]) - 0.40 * dy          # housing rises towards the outer corner
            d = np.sqrt(((X3 - p[0]) ** 2 + dy ** 2) / r ** 2 + (dz / (0.30 * r)) ** 2)
            put(d < 1, C['black'], 0.9)
    # roof
    put(J2 >= rj.J_EAVE + 2.5, C['roof'], 0.3)
    put(J2 < 2.5, C['under'], 0.15)
    # window bands
    def band(rows, x0, x1):
        cells = np.isin(np.floor(J2).astype(int), rows)
        return cells & (X >= x0) & (X <= x1)
    up = [w for w in spec.windows if w[2] == rj.CELLS_UPPER_WIN]
    lo = [w for w in spec.windows if w[2] == rj.CELLS_LOWER_WIN]
    if up:
        m = (X >= up[0][0] - 0.18) & (X <= up[-1][1] + 0.18) & (Z > 2.92) & (J2 < rj.J_EAVE + 0.4)
        put(m, C['black'], 0.85)
    if lo:
        m = (X >= lo[0][0] - 0.12) & (X <= lo[-1][1] + 0.12) & (Z > 0.88) & (Z < 1.92)
        put(m, C['black'], 0.85)
    # individual windows: painted glass (used by LOD 1/2) + gasket
    for (a, b, cells) in spec.windows:
        if cells == rj.CELLS_WINDSCREEN:
            m = (X >= a) & (X <= b) & (J2 >= min(cells)) & (J2 <= max(cells) + 1)
            put(m, C['glass'], 0.95)
            continue
        jc = np.floor(J2).astype(int)
        zs = Z[(jc >= min(cells)) & (jc <= max(cells)) & (np.abs(X - (a + b) / 2) < 0.01)]
        if zs.size == 0:
            continue
        z0, z1 = zs.min(), zs.max()
        d = _roundrect_d(X, Z, a, b, z0, z1, 0.10)
        put(d < 0.035, C['black'], 0.6)
        g = d < 0
        shade = np.clip((Z - z0) / max(z1 - z0, 1e-3), 0, 1)[..., None]
        col[g] = (np.array(C['glass']) * (0.8 + 0.5 * shade))[g]
        gl[g] = 0.95
        hgt[(d > 0) & (d < 0.035)] = 0.5
    # doors
    stamp_px = int(0.16 * px / 27.2 * 1.0 * 6)
    for (a, b, cells) in spec.doors:
        d_out = _roundrect_d(X, Z, a - 0.11, b + 0.11, 0.40, 2.74, 0.10)
        d_in = _roundrect_d(X, Z, a, b, 0.50, 2.64, 0.02)
        put((d_out < 0) & (d_in >= 0) & (J2 > 2.5), C['frame'], 0.5)
        hgt[(np.abs(d_in) < 0.012)] = -1
        inside = (X > a) & (X < b) & (J2 > 4.5) & (Z < 2.64)
        put(inside, C['door'], 0.5)
        c = (a + b) / 2
        put(inside & (Z > 2.38), C['black'], 0.8)
        for lc in (c - spec.door_w / 4, c + spec.door_w / 4):
            dw = _roundrect_d(X, Z, lc - 0.16, lc + 0.16, 1.30, 2.25, 0.07)
            put(dw < 0, C['glass'], 0.95)
            put((dw >= 0) & (dw < 0.02), C['black'])
        put(inside & (np.abs(X - c) < 0.006), C['black'])
        hgt[inside & (np.abs(X - c) < 0.006)] = -1
    # seams between body modules
    for xseam in (spec.low[0] - 0.6, spec.low[1] + 0.6):
        m = (np.abs(X - xseam) < 0.008) & (J2 > 3) & (J2 < rj.J_EAVE)
        col[m] *= 0.6
        hgt[m] = -1
    # ambient occlusion & dirt
    ao *= np.clip(0.75 + 0.25 * (Z - 0.3) / 0.6, 0.75, 1)
    dmask = np.clip((1.3 - Z) / 1.0, 0, 1) * 150 + np.clip(J2 - rj.J_EAVE, 0, 3) * 20 + 25
    atlas[r0:r1] = col
    mga[r0:r1, :, 0] = met * 255
    mga[r0:r1, :, 1] = gl * 255
    mga[r0:r1, :, 2] = ao * 255
    height[r0:r1] = hgt
    dirt[r0:r1] = dmask
    # class numbers next to / on the doors and small lettering (texel space)
    for (a, b, cells) in spec.doors:
        c = (a + b) / 2
        cxp = (c + 13.6) / 27.2 * px
        jtop = np.interp(2.51, Z[:, int(cxp)][::-1], np.arange(R)[::-1]) if Z[0, int(cxp)] > Z[-1, int(cxp)] else None
        rows = np.where((Z[:, int(cxp)] > 2.40) & (Z[:, int(cxp)] < 2.62))[0]
        if len(rows):
            cy = r0 + rows.mean()
            st = _text_stamp(cls, max(10, int(0.15 * px / 27.2)), (255, 255, 255) if cls == '2' else (250, 200, 0))
            _paste(atlas, st, cxp, cy)
    for x in ((spec.low[0] + 0.8), (spec.low[1] - 2.2)):
        cxp = (x + 13.6) / 27.2 * px
        rows = np.where((Z[:, int(cxp)] > 0.62) & (Z[:, int(cxp)] < 0.72))[0]
        if len(rows):
            st = _text_stamp('93 81 4' + ('011' if spec.kind == 'end' else '211') + ' 001-7', max(8, int(0.07 * px / 27.2)), (220, 220, 220))
            _paste(atlas, st, cxp, r0 + rows.mean())


def normal_from_height(h, strength=2.0):
    gy, gx = np.gradient(h)
    n = np.stack([-gx * strength, -gy * strength, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    return np.stack([n[..., 0] * 127 + 128, n[..., 1] * 127 + 128], -1)


def dilate(img, valid, iters=8):
    """Push chart colours into unused texels to avoid seams at mip levels."""
    img = img.copy()
    v = valid.copy()
    for _ in range(iters):
        acc = np.zeros_like(img)
        cnt = np.zeros(v.shape)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sv = np.roll(v, (dy, dx), (0, 1))
            si = np.roll(img, (dy, dx), (0, 1))
            acc[sv & ~v] += si[sv & ~v]
            cnt[sv & ~v] += 1
        new = (cnt > 0) & ~v
        img[new] = acc[new] / cnt[new][:, None]
        v |= new
    return img


def glass_texture(px=512):
    """Left half: side windows, right half: windscreen. RGBA."""
    img = np.zeros((px, px, 4))
    yy, xx = np.mgrid[0:px, 0:px] + 0.5
    for (u0, u1, alpha, r) in ((0, px // 2, 190, 0.16), (px // 2, px, 175, 0.08)):
        w = u1 - u0
        u = (xx - u0) / w
        v = yy / px
        d = _roundrect_d(u, v, 0, 1, 0, 1, r) * 1.0
        reg = (xx >= u0) & (xx < u1)
        tint = np.array([62, 88, 112.0]) * (0.75 + 0.6 * (1 - v))[..., None]
        img[reg, :3] = tint[reg]
        img[reg, 3] = alpha
        gasket = reg & (d > -0.035)
        img[gasket, :3] = 14
        img[gasket, 3] = 255
    return img


def interior_texture(sw, px):
    rgb = np.zeros((px, px, 3))
    mga = np.zeros((px, px, 3))
    sw.paint(rgb, mga)
    return rgb, mga

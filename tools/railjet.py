"""Geometry of the OeBB Railjet double-deck (Stadler KISS based) cars.

Every car is built with its driving cab (if any) at +X. The rear driving trailer is
created later by rotating the finished geometry by 180 degrees around Z.

Dimensions follow published figures where they exist (6 cars, ~160 m, 200 km/h,
4.6 m high, 2.8 m wide); everything else (zone lengths, window pitch, bogie
positions, ...) is estimated from photos of KISS double-deck trains.
"""
import math
import numpy as np

from curves import pchip, smoothstep
from meshlib import Mesh, box, cylinder, disc_ring, translate, rot_z, rot_y, rot_x

# ----------------------------------------------------------------------------- profile
HALF_W = 1.40
TOP = 4.60
EAVE = 4.05
LEVELS = [0.55, 0.95, 1.75, 1.90, 2.62, 2.80, 3.00, 3.92]   # j = 5 .. 12
J_EAVE = 13
ROOF_N = 7                                                    # j = 14 .. 20
J = 22                                                        # last j = 21 is roof centre
ZB_LOW = 0.38
ZB_HIGH = 1.05
FLOOR_LOW = 0.62
FLOOR_MEZZ = 1.30
FLOOR_UP = 2.68
CEIL_LOW = 2.52
SHELL_T = 0.07
ROOF_END = 4.20

# rows (cells between j and j+1) used by openings
CELLS_LOWER_WIN = (6, 7)          # 0.95 .. 1.90
CELLS_MEZZ_WIN = (7, 8, 9)        # 1.75 .. 2.80
CELLS_UPPER_WIN = (11,)           # 3.00 .. 3.92
CELLS_DOOR = (5, 6, 7, 8)         # 0.55 .. 2.62
CELLS_CABWIN = (10, 11)
CELLS_WINDSCREEN = (9, 10, 11, 12)
CELLS_RECESS = (6,)

# driving cab nose (end cars). Nominal stations XS..XT are a parameter t = 0..1 that is the
# same "feature line" on every profile row j:
#   t in [0, T_CORNER]       straight side wall
#   t in [T_CORNER, T_FRONT] corner arc (radius NOSE_R[j])
#   t in [T_FRONT, 1]        slightly curved front (radius NOSE_RF[j]) to the centre line
# In side view every row j has its own height NOSE_Z[j] and front position NOSE_X[j]; these
# points lie on straight lines (skirt, chin, shelf, windscreen) with an elliptic roof hood.
XS = 10.0
XT = 13.28
T_CORNER = 0.30
T_FRONT = 0.62
CAB_TOP = TOP
CHIN_STEP = 0.035          # cab sides above the headlight crease are slightly narrower
#          j:   0      1      2      3      4      5      6      7      8      9      10      11     12      13
NOSE_Z = [0.30, 0.30, 0.30, 0.33, 0.42, 0.80, 0.95, 1.70, 2.15, 2.45, 2.875, 3.30, 3.725, 4.15]
NOSE_X = [12.95, 12.95, 12.95, 12.98, 13.05, 13.16, 13.28, 13.28, 13.25, 12.75, 12.475, 12.20, 11.925, 11.65]
HOOD_X0, HOOD_X1 = 10.85, 0.80     # roof hood: x = X0 + X1 cos(phi), z = 4.15 + 0.45 sin(phi)
NOSE_CREASES = (2, 5, 6, 8, 9, 13)


def nose_rows(base):
    """Per-row target (W, z, xf, R, Rf) of the nose, from the cab section at XS."""
    W = base[:, 0].copy()
    z = base[:, 1].copy()
    xf = np.zeros(J)
    R = np.zeros(J)
    Rf = np.zeros(J)
    for j in range(J):
        if j <= J_EAVE:
            z[j] = NOSE_Z[j]
            xf[j] = NOSE_X[j]
        else:
            phi = (j - J_EAVE) / (J - 1 - J_EAVE) * math.pi / 2
            z[j] = NOSE_Z[J_EAVE] + (CAB_TOP - NOSE_Z[J_EAVE]) * math.sin(phi)
            xf[j] = HOOD_X0 + HOOD_X1 * math.cos(phi)
        if j >= 9:
            W[j] = max(W[j] - CHIN_STEP, 0.0) if W[j] > 0 else 0.0
        if j <= 4:
            R[j], Rf[j] = 0.50, 4.0
        elif j <= 8:
            R[j], Rf[j] = 0.45, 4.0
        elif j <= J_EAVE:
            R[j], Rf[j] = 0.30, 3.0
        else:
            R[j], Rf[j] = min(0.30, 0.8 * W[j]), 3.0
    return W, z, xf, R, Rf


def plan_point(t, W, xf, R, Rf):
    """Plan-view point (x, y) of one row at parameter t (array)."""
    t = np.asarray(t, float)
    if W < 1e-4:
        return XS + (xf - XS) * np.clip(t, 0, 1), np.zeros_like(t)
    R = min(R, W)
    xc = xf - Rf + math.sqrt((Rf - R) ** 2 - (W - R) ** 2)
    xc = max(xc, XS + 0.02)
    dvec = np.array([xc - (xf - Rf), W - R])
    dvec /= np.linalg.norm(dvec)
    a_t = math.atan2(dvec[0], dvec[1])        # corner angle measured from +y towards +x
    b_t = math.atan2(dvec[1], dvec[0])        # front arc angle measured from +x
    x = np.empty_like(t)
    y = np.empty_like(t)
    s = t < T_CORNER
    x[s] = XS + (xc - XS) * t[s] / T_CORNER
    y[s] = W
    c = (t >= T_CORNER) & (t < T_FRONT)
    al = a_t * (t[c] - T_CORNER) / (T_FRONT - T_CORNER)
    x[c] = xc + R * np.sin(al)
    y[c] = (W - R) + R * np.cos(al)
    f = t >= T_FRONT
    be = b_t * (1 - (np.clip(t[f], 0, 1) - T_FRONT) / (1 - T_FRONT))
    x[f] = xf - Rf + Rf * np.cos(be)
    y[f] = Rf * np.sin(be)
    return x, y


def t_to_x(t):
    return XS + t * (XT - XS)


def side_w(z):
    z = np.asarray(z, float)
    w = np.full_like(z, HALF_W)
    lo = z < 0.75
    w = np.where(lo, HALF_W - 0.012 * np.clip((0.75 - z) / 0.4, 0, 1), w)
    hi = z > 3.0
    w = np.where(hi, HALF_W - 0.07 * np.clip((z - 3.0) / (EAVE - 3.0), 0, 1) ** 1.6, w)
    return w


def base_profile(zb):
    """Half profile (y >= 0) for body underside height zb. zb: array (N,). Returns (N, J, 2)."""
    zb = np.asarray(zb, float)
    N = zb.shape[0]
    P = np.zeros((N, J, 2))
    P[:, 0] = np.stack([np.zeros(N), zb], -1)
    P[:, 1] = np.stack([np.full(N, 0.80), zb], -1)
    P[:, 2] = np.stack([np.full(N, 1.27), zb], -1)
    P[:, 3] = np.stack([np.full(N, 1.355), zb + 0.045], -1)
    P[:, 4] = np.stack([side_w(zb + 0.15), zb + 0.15], -1)
    for k, L in enumerate(LEVELS):
        z = np.maximum(L, zb + 0.15 + 0.001 * (k + 1))
        P[:, 5 + k] = np.stack([side_w(z), z], -1)
    we = side_w(np.array(EAVE))
    P[:, J_EAVE] = np.stack([np.full(N, we), np.full(N, EAVE)], -1)
    for k in range(ROOF_N):
        a = (k + 1) / (ROOF_N + 1) * math.pi / 2
        y = we * math.cos(a) ** 0.62
        z = EAVE + (TOP - EAVE) * math.sin(a) ** 0.85
        P[:, J_EAVE + 1 + k] = (y, z)
    P[:, J - 1] = (0, TOP)
    return P


def nominal_v():
    """Normalised arclength of the low-zone profile, used as texture v for every row j."""
    P = base_profile(np.array([ZB_LOW]))[0]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    return s / s[-1]


VNOM = nominal_v()


# ----------------------------------------------------------------------------- car specs
def window_row(a, b, w, pier):
    n = int((b - a + pier) // (w + pier))
    span = n * (w + pier) - pier
    x = a + (b - a - span) / 2
    out = []
    for _ in range(n):
        out.append((x, x + w))
        x += w + pier
    return out


class CarSpec:
    def __init__(self, kind):
        self.kind = kind
        if kind == 'mid':
            self.L = 26.5
            self.x_rear = -13.0
            self.x_front = 13.0
            self.bogies = [-9.6, 9.6]
            self.low = (-7.0, 7.0)
            self.door_xs = [-6.2, 6.2]
            lw = window_row(-5.40, 5.40, 1.25, 0.45)
            uw = window_row(-6.85, 6.85, 1.25, 0.45)
            mz = window_row(7.85, 12.45, 1.30, 0.75)
            self.windows = ([(a, b, CELLS_LOWER_WIN) for a, b in lw] +
                            [(a, b, CELLS_UPPER_WIN) for a, b in uw] +
                            [(a, b, CELLS_MEZZ_WIN) for a, b in mz] +
                            [(-b, -a, CELLS_MEZZ_WIN) for a, b in mz])
        else:
            self.L = 27.0
            self.x_rear = -13.25
            self.x_front = 13.25
            self.bogies = [-9.85, 8.3]
            self.low = (-7.25, 5.6)
            self.door_xs = [-6.05, 4.35]
            lw = window_row(-4.95, 3.25, 1.25, 0.45)
            uw = window_row(-6.70, 4.95, 1.25, 0.45)
            mz = [(-10.45, -9.20)]
            self.windows = ([(a, b, CELLS_LOWER_WIN) for a, b in lw] +
                            [(a, b, CELLS_UPPER_WIN) for a, b in uw] +
                            [(a, b, CELLS_MEZZ_WIN) for a, b in mz] +
                            [(7.00, 7.60, CELLS_MEZZ_WIN), (t_to_x(0.08), t_to_x(0.22), CELLS_CABWIN)] +
                            [(t_to_x(0.46), XT + 0.01, CELLS_WINDSCREEN)])
        self.recess = [(t_to_x(0.74), XT + 0.01, CELLS_RECESS)] if kind == 'end' else []
        self.door_w = 1.30
        self.doors = [(x - self.door_w / 2, x + self.door_w / 2, CELLS_DOOR) for x in self.door_xs]
        self.door_x = self.door_xs[0]
        self.half_len = self.L / 2

    def ztop(self, x):
        """Roof height: full height over the double-deck zone, lowered over the bogies."""
        x = np.asarray(x, float)
        a, b = self.low
        t = smoothstep(a - 0.9, a - 0.2, x) * (1 - smoothstep(b + 0.2, b + 0.9, x))
        if self.kind == 'end':
            t = smoothstep(a - 0.9, a - 0.2, x)
        return ROOF_END + (TOP - ROOF_END) * t

    # --- sections ------------------------------------------------------------
    def zb(self, x):
        x = np.asarray(x, float)
        a, b = self.low
        r = 0.4
        t = smoothstep(a - r, a, x) * (1 - smoothstep(b, b + r, x))
        z = ZB_HIGH + (ZB_LOW - ZB_HIGH) * t
        if self.kind == 'end':
            z = z - 0.20 * smoothstep(9.3, 10.0, x)
        return z

    def section(self, x):
        """Half profile at stations x (array) -> (N, J, 2)."""
        x = np.atleast_1d(np.asarray(x, float))
        P = base_profile(self.zb(x))
        ztop = self.ztop(x)
        zp = 1.45
        z = P[..., 1]
        f = ((ztop - zp) / (TOP - zp))[:, None]
        P[..., 1] = np.where(z > zp, zp + (z - zp) * f, z)
        return P

    def section3d(self, x):
        """3D points (N, J, 3) for nominal stations x. Equal to (x, section) except in the nose."""
        x = np.atleast_1d(np.asarray(x, float))
        S = self.section(np.minimum(x, XS) if self.kind == 'end' else x)
        out = np.zeros(S.shape[:2] + (3,))
        out[..., 0] = x[:, None]
        out[..., 1] = S[..., 0]
        out[..., 2] = S[..., 1]
        if self.kind != 'end':
            return out
        nose = x > XS
        if not nose.any():
            return out
        base = self.section(np.array([XS]))[0]                      # (J, 2)
        W, zt, xf, R, Rf = nose_rows(base)
        t = np.clip((x[nose] - XS) / (XT - XS), 0, 1)
        # skirt rows only drop down around the front corners; all other rows settle early
        skirt = np.arange(J) <= 5
        blend = np.where(skirt[None, :], smoothstep(0.22, 0.56, t)[:, None], smoothstep(0.0, 0.15, t)[:, None])
        X = np.zeros((len(t), J))
        Y = np.zeros((len(t), J))
        for j in range(J):
            X[:, j], Y[:, j] = plan_point(t, W[j], xf[j], R[j], Rf[j])
        # rows start at the cab section and reach their nose width/height early on the straight part
        Y = np.where(t[:, None] < T_CORNER, base[None, :, 0] + (Y - base[None, :, 0]) * np.minimum(blend, 1), Y)
        out[nose, :, 0] = X
        out[nose, :, 1] = Y
        out[nose, :, 2] = base[None, :, 1] + (zt[None, :] - base[None, :, 1]) * blend
        return out

    def stations(self, lod):
        pts = {self.x_rear, self.x_front}
        a, b = self.low
        for e in (a, b):
            for k in range(5):
                pts.add(e - 0.4 + 0.1 * k)
        for x0, x1, _ in self.windows + self.doors:
            pts.update((x0, x1))
        pts.update(self.door_xs)
        for e in (a, b):
            pts.update((e - 0.9, e - 0.55, e - 0.2, e + 0.2, e + 0.55, e + 0.9))
        nose_t = {0: [0, .05, .08, .10, .15, .20, .22, .25, .27, .30, .34, .38, .42, .46, .50, .54, .58, .62,
                      .66, .70, .74, .78, .82, .86, .90, .94, .97, 1.0],
                  1: [0, .08, .15, .22, .30, .38, .46, .54, .62, .74, .86, 1.0],
                  2: [0, .30, .46, .62, .80, 1.0]}[lod]
        if self.kind == 'end':
            pts = {p for p in pts if p <= XS}
            pts.update((9.3, 9.5, 9.7, 9.85))
            pts.update(t_to_x(t) for t in nose_t)
        xs = sorted(pts)
        maxd = {0: 0.45, 1: 0.9, 2: 2.0}[lod]
        if lod == 2:
            keep = {self.x_rear, self.x_front, a - 0.4, a, b, b + 0.4}
            if self.kind == 'end':
                keep = {p for p in keep if p <= XS} | {9.3, 9.7} | {t_to_x(t) for t in nose_t}
            xs = sorted(keep)
        out = [xs[0]]
        for x in xs[1:]:
            if x - out[-1] < 0.004:
                continue
            n = int(math.ceil((x - out[-1]) / maxd)) if x <= XS or self.kind != 'end' else 1
            last = out[-1]
            for k in range(1, n):
                out.append(last + (x - last) * k / n)
            out.append(x)
        return np.array(out)

    def opening(self, x0, x1, j):
        """Return ('win', idx) / ('door', idx) / None for the cell between stations x0, x1, row j."""
        xc = 0.5 * (x0 + x1)
        for i, (a, b, cells) in enumerate(self.doors):
            if a <= xc <= b and j in cells:
                return ('door', i)
        for i, (a, b, cells) in enumerate(self.windows):
            if a <= xc <= b and j in cells:
                return ('win', i)
        for i, (a, b, cells) in enumerate(self.recess):
            if a <= xc <= b and j in cells:
                return ('recess', i)
        return None


# ----------------------------------------------------------------------------- loft
class Loft:
    """Evaluated body surface for one car and LOD."""

    def __init__(self, spec, lod, chart):
        self.spec = spec
        self.lod = lod
        self.chart = chart  # (v0, v1) rows of the body atlas, u spans 0..1 for x in [-13.6, 13.6]
        self.xs = spec.stations(lod)
        I = len(self.xs)
        P = spec.section3d(self.xs)
        self.P = P
        dx = np.gradient(P, axis=0)
        dj = np.gradient(P, axis=1)
        N = np.cross(dj, dx)
        nl = np.linalg.norm(N, axis=2, keepdims=True)
        N = np.where(nl > 1e-9, N / np.maximum(nl, 1e-12), np.array([0, 0, 1.0]))
        # vertices on the mirror plane (y = 0) must not have a sideways normal component
        on_seam = np.abs(P[..., 1]) < 1e-6
        N[..., 1] = np.where(on_seam, 0.0, N[..., 1])
        N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-12
        self.N = N
        # one-sided normals for crease rows of the cab nose (sharp feature lines)
        self.crease = set(NOSE_CREASES) if spec.kind == 'end' else set()
        self.in_nose = (self.xs >= XS - 1e-6) if spec.kind == 'end' else np.zeros(I, bool)
        dup = np.zeros_like(P)
        dn = np.zeros_like(P)
        dup[:, :-1] = P[:, 1:] - P[:, :-1]
        dup[:, -1] = dup[:, -2]
        dn[:, 1:] = P[:, 1:] - P[:, :-1]
        dn[:, 0] = dn[:, 1]

        def norm_of(dj_):
            n = np.cross(dj_, dx)
            n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
            n[..., 1] = np.where(on_seam, 0.0, n[..., 1])
            return n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-12)
        self.N_up = norm_of(dup)     # use for a crease row as the lower edge of a cell
        self.N_dn = norm_of(dn)      # use for a crease row as the upper edge of a cell
        self.U = np.zeros((I, J, 2))
        self.U[..., 0] = ((self.xs + 13.6) / 27.2)[:, None]
        self.U[..., 1] = (chart[0] + VNOM * (chart[1] - chart[0]))[None, :]

    def cell_open(self, i, j):
        if self.lod == 2:
            return None
        o = self.spec.opening(self.xs[i], self.xs[i + 1], j)
        if o is None:
            return None
        if self.lod == 1 and o[0] != 'door':
            return None
        return o

    def shade_normal(self, a, b, cell_j):
        if b in self.crease and self.in_nose[a]:
            return (self.N_up if b == cell_j else self.N_dn)[a, b].copy()
        return self.N[a, b].copy()

    def corners(self, i, j, side, offset=0.0):
        idx = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
        pts, nrm, uvs = [], [], []
        for a, b in idx:
            p = self.P[a, b] - offset * self.N[a, b]
            n = self.shade_normal(a, b, j)
            if side < 0:
                p = p * (1, -1, 1)
                n = n * (1, -1, 1)
            pts.append(p)
            nrm.append(n)
            uvs.append(self.U[a, b])
        if side < 0:
            pts, nrm, uvs = pts[::-1], nrm[::-1], uvs[::-1]
        return pts, nrm, uvs

    def build_shell(self, mesh, slot, paint='body', holes=True):
        I = len(self.xs)
        for side in (1, -1):
            for i in range(I - 1):
                for j in range(J - 1):
                    if holes and self.cell_open(i, j):
                        continue
                    pts, nrm, uvs = self.corners(i, j, side)
                    h = np.linalg.norm(self.P[i, j + 1] - self.P[i, j]) + np.linalg.norm(self.P[i + 1, j + 1] - self.P[i + 1, j])
                    if h < 0.003:
                        continue
                    flat = j < 2
                    mesh.quad(pts, uvs, None if flat else nrm, slot, paint)

    def build_inner(self, mesh, slot, uv, xlim):
        """Inner wall surface (faces inwards) for j >= 4 between xlim."""
        I = len(self.xs)
        for side in (1, -1):
            for i in range(I - 1):
                if self.xs[i] < xlim[0] - 1e-6 or self.xs[i + 1] > xlim[1] + 1e-6:
                    continue
                for j in range(4, J - 1):
                    if self.cell_open(i, j):
                        continue
                    pts, nrm, _ = self.corners(i, j, side, SHELL_T)
                    mesh.quad(pts[::-1], uv('f', None), [-n for n in nrm[::-1]], slot, 'wall')

    def build_reveals(self, mesh, slot, uv_gasket, uv_wall, xlim):
        I = len(self.xs)
        tip = self.spec.kind == 'end'
        for side in (1, -1):
            for i in range(I - 1):
                for j in range(J - 1):
                    o = self.cell_open(i, j)
                    if not o:
                        continue
                    depth = 0.42 if o[0] == 'recess' else SHELL_T
                    if o[0] == 'recess':
                        # back wall of the coupler recess
                        pb, nb, _ = self.corners(i, j, side, depth)
                        mesh.quad(pb, [uv_gasket('f', None)[0]] * 4, None, slot, 'gasket')
                    # neighbours: (di, dj, edge corner indices in the order of corners())
                    for di, dj, e in ((-1, 0, (0, 1)), (1, 0, (2, 3)), (0, -1, (3, 0)), (0, 1, (1, 2))):
                        ni, nj = i + di, j + dj
                        if tip and di == 1 and ni == I - 1:
                            continue        # opening continues on the mirrored half
                        inside = 0 <= ni < I - 1 and 0 <= nj < J - 1
                        if inside and self.cell_open(ni, nj):
                            continue
                        po, no, _ = self.corners(i, j, 1)
                        pi_, _, _ = self.corners(i, j, 1, depth)
                        a, b = e
                        q = [po[a], po[b], pi_[b], pi_[a]]
                        if side < 0:
                            q = [p * (1, -1, 1) for p in q][::-1]
                        uvq = uv_gasket('f', None)
                        mesh.quad(q, uvq, None, slot, 'gasket')

    def raycast(self, y, z, xmin=9.5):
        """Front-most intersection of a ray along -x at (y, z) with the outer shell: (point, normal)."""
        if not hasattr(self, '_tris'):
            m = Mesh(1)
            self.build_shell(m, 0, holes=False)
            P, _, _ = m.arrays()
            keep = P[:, :, 0].max(1) > xmin
            self._tris = P[keep]
        T = self._tris
        o = np.array([40.0, y, z])
        d = np.array([-1.0, 0, 0])
        e1 = T[:, 1] - T[:, 0]
        e2 = T[:, 2] - T[:, 0]
        h = np.cross(d, e2)
        a = (e1 * h).sum(1)
        ok = np.abs(a) > 1e-12
        f = np.where(ok, 1 / np.where(ok, a, 1), 0)
        s = o - T[:, 0]
        u = f * (s * h).sum(1)
        q = np.cross(s, e1)
        v = f * (q @ d)
        t = f * (e2 * q).sum(1)
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
        if not hit.any():
            return None
        k = np.where(hit)[0][np.argmin(t[hit])]
        n = np.cross(e1[k], e2[k])
        n /= np.linalg.norm(n)
        return o + d * t[k], n

    def glass(self, mesh, slot, regions):
        """Window glass patches; regions: kind -> (u0, v0, u1, v1) of the glass texture."""
        spec = self.spec
        for wi, (a, b, cells) in enumerate(spec.windows):
            region = regions['windscreen' if cells == CELLS_WINDSCREEN else 'window']
            ii = [i for i in range(len(self.xs) - 1) if a <= 0.5 * (self.xs[i] + self.xs[i + 1]) <= b]
            if not ii:
                continue
            j0, j1 = min(cells), max(cells) + 1
            for side in (1, -1):
                for i in ii:
                    s = []
                    for k in (i, i + 1):
                        seg = np.linalg.norm(np.diff(self.P[k, j0:j1 + 1], axis=0), axis=1)
                        s.append(np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9))
                    for j in cells:
                        idx = [(i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)]
                        pts, nrm, uvs = [], [], []
                        for (pi_, pj) in idx:
                            p = self.P[pi_, pj] - 0.035 * self.N[pi_, pj]
                            n = self.N[pi_, pj].copy()
                            fu = (self.xs[pi_] - a) / (b - a)
                            if cells == CELLS_WINDSCREEN:
                                fu = 0.5 * min(fu, 1.0)
                            fv = 1 - s[pi_ - i][pj - j0]
                            u = region[0] + fu * (region[2] - region[0])
                            v = region[1] + fv * (region[3] - region[1])
                            if side < 0:
                                p = p * (1, -1, 1)
                                n = n * (1, -1, 1)
                            pts.append(p)
                            nrm.append(n)
                            uvs.append((u, v))
                        if side < 0:
                            pts, nrm, uvs = pts[::-1], nrm[::-1], uvs[::-1]
                        mesh.quad(pts, uvs, nrm, slot, 'glass')

    def door_leaves(self, door_idx, side, leaf):
        """Mesh of one door leaf (leaf -1: rear half, +1: front half) in model coordinates."""
        a, b, cells = self.spec.doors[door_idx]
        c = 0.5 * (a + b)
        m = Mesh(1)
        ii = [i for i in range(len(self.xs) - 1)
              if (a <= self.xs[i] and self.xs[i + 1] <= c + 1e-6 if leaf < 0 else c - 1e-6 <= self.xs[i] and self.xs[i + 1] <= b)]
        j0, j1 = min(cells), max(cells) + 1
        for i in ii:
            for j in cells:
                pts, nrm, uvs = self.corners(i, j, side, -0.004)
                m.quad(pts, uvs, nrm, 0, 'door')
                pin, nin, _ = self.corners(i, j, side, 0.05)
                m.quad(pin[::-1], [uvs[0]] * 4, [-n for n in nin[::-1]], 0, 'door_in')
        # edges
        x0 = self.xs[ii[0]]
        x1 = self.xs[ii[-1] + 1]
        for xi in (ii[0], ii[-1] + 1):
            for j in cells:
                p0 = self.P[xi, j] * (1, side, 1)
                p1 = self.P[xi, j + 1] * (1, side, 1)
                n0 = self.N[xi, j] * (1, side, 1)
                n1 = self.N[xi, j + 1] * (1, side, 1)
                q = [p0 + 0.004 * n0, p1 + 0.004 * n1, p1 - 0.05 * n1, p0 - 0.05 * n0]
                fwd = xi != ii[0]
                if (side > 0) == fwd:
                    q = q[::-1]
                m.quad(q, [self.U[xi, j]] * 4, None, 0, 'door_edge')
        for jj, top in ((j0, False), (j1, True)):
            for i in ii:
                p0 = self.P[i, jj] * (1, side, 1)
                p1 = self.P[i + 1, jj] * (1, side, 1)
                n0 = self.N[i, jj] * (1, side, 1)
                n1 = self.N[i + 1, jj] * (1, side, 1)
                q = [p0 + 0.004 * n0, p1 + 0.004 * n1, p1 - 0.05 * n1, p0 - 0.05 * n0]
                if (side > 0) != top:
                    q = q[::-1]
                m.quad(q, [self.U[i, jj]] * 4, None, 0, 'door_edge')
        return m, (x0, x1)

    def end_cap(self, mesh, slot, x_end, facing, uv):
        """Flat end wall at the station closest to x_end."""
        i = int(np.argmin(np.abs(self.xs - x_end)))
        prof = self.P[i]
        cen = np.array([self.xs[i], 0, 2.6])
        full = list(prof) + [p * (1, -1, 1) for p in prof[::-1][1:-1]]
        n = len(full)
        for k in range(n):
            a = full[k]
            b = full[(k + 1) % n]
            tri = [cen, a, b]
            # orientation: ensure normal points along facing
            nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            if nrm[0] * facing < 0:
                tri = [cen, b, a]
            mesh.tri(tri, uv('cap', (0, 0, 0, 0)), None, slot, 'endwall')

    def surface_point(self, y, z):
        return self.raycast(y, z)

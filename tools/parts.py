"""Detail parts: wheelsets, bogies, pantograph, gangways, couplers, roof equipment,
lights and the passenger interior."""
import math
import numpy as np

from meshlib import Mesh, box, cylinder, disc_ring, translate, rot_z, rot_y, rot_x
import railjet as rj

WHEEL_R = 0.46
AXLE_Z = WHEEL_R
WHEELBASE = 2.6


def lathe(mesh, prof, seg, uvf, axis_y=True, slot=0, paint=None):
    """Revolve a (r, y) polyline around the Y axis. Normals smooth around, flat along profile."""
    for k in range(len(prof) - 1):
        r0, y0 = prof[k]
        r1, y1 = prof[k + 1]
        for i in range(seg):
            a0 = 2 * math.pi * i / seg
            a1 = 2 * math.pi * (i + 1) / seg
            p = [(r0 * math.cos(a0), y0, r0 * math.sin(a0)), (r0 * math.cos(a1), y0, r0 * math.sin(a1)),
                 (r1 * math.cos(a1), y1, r1 * math.sin(a1)), (r1 * math.cos(a0), y1, r1 * math.sin(a0))]
            mesh.quad(p, uvf('f', None), None, slot, paint)


def wheelset(sw, seg, discs):
    """Wheelset centred on its axle (origin at axle centre)."""
    steel = sw.uvf('wheel_steel')
    tread = sw.uvf('wheel_tread')
    dark = sw.uvf('bogie_dark')
    w = Mesh(1)
    # wheel cross-section in (r, y), counter-clockwise; revolved clockwise -> outward normals
    ccw = [(0.11, 0.65), (0.20, 0.65), (0.38, 0.68), (WHEEL_R + 0.02, 0.68), (WHEEL_R + 0.028, 0.70),
           (WHEEL_R + 0.002, 0.715), (WHEEL_R, 0.81), (0.40, 0.81), (0.36, 0.79), (0.20, 0.82), (0.11, 0.86)]
    cw = ccw[::-1]
    for k in range(len(cw) - 1):
        uv = tread if 4 <= len(cw) - 2 - k <= 5 else steel
        lathe(w, cw[k:k + 2], seg, uv)
    if discs:
        for yd in (0.42, 0.20):
            cylinder(w, (0, yd - 0.05, 0), (0, yd + 0.05, 0), 0.31, seg, dark)
    m = Mesh(1)
    m.extend(w)
    m.extend(w, np.diag([1.0, -1.0, 1.0, 1.0]))
    cylinder(m, (0, -0.86, 0), (0, 0.86, 0), 0.085, max(8, seg // 2), steel)
    return m


def bogie(sw, motor, lod):
    """Bogie frame centred at its pivot, z = 0 at rail top."""
    m = Mesh(1)
    seg = 12 if lod == 0 else 8
    frame = sw.uvf('bogie_frame')
    dark = sw.uvf('bogie_dark')
    rubber = sw.uvf('rubber')
    yellow = sw.uvf('axlebox_cover')
    spring = sw.uvf('spring')
    hb = WHEELBASE / 2
    for s in (1, -1):
        y0, y1 = 0.93 * s, 1.11 * s
        lo, hi = (min(y0, y1), max(y0, y1))
        # side frame: ends over the axle boxes, dropped centre
        box(m, (-hb - 0.45, lo, 0.62), (-hb + 0.35, hi, 0.86), frame)
        box(m, (hb - 0.35, lo, 0.62), (hb + 0.45, hi, 0.86), frame)
        box(m, (-hb + 0.35, lo, 0.50), (hb - 0.35, hi, 0.78), frame)
        # axle boxes + covers
        for ax in (-hb, hb):
            cylinder(m, (ax, 0.86 * s, AXLE_Z), (ax, 1.06 * s, AXLE_Z), 0.15, seg, dark)
            cylinder(m, (ax, 1.06 * s, AXLE_Z), (ax, 1.12 * s, AXLE_Z), 0.12, seg, yellow)
            if lod == 0:
                for dx in (-0.22, 0.22):
                    cylinder(m, (ax + dx, 1.02 * s, 0.52), (ax + dx, 1.02 * s, 0.66), 0.075, 8, spring, caps=False)
        # secondary air springs
        cylinder(m, (0, 1.02 * s, 0.78), (0, 1.02 * s, 1.02), 0.22, seg, rubber)
        if lod == 0:
            # yaw damper and vertical damper
            cylinder(m, (-0.95, 1.16 * s, 0.84), (0.25, 1.16 * s, 0.84), 0.045, 8, dark)
            cylinder(m, (0.55, 1.13 * s, 0.55), (0.65, 1.13 * s, 0.95), 0.04, 8, dark)
            # magnetic track brake (trailer) / sander
            if not motor:
                box(m, (-hb + 0.5, 0.70 * s - 0.06, 0.12), (hb - 0.5, 0.70 * s + 0.06, 0.24), dark)
                for ax in (-hb, hb):
                    box(m, (ax - 0.12, 0.40 * s - 0.08, 0.25), (ax + 0.12, 0.40 * s + 0.08, 0.62), dark)
            else:
                cylinder(m, (-hb - 0.40, 0.75 * s, 0.25), (-hb - 0.40, 0.75 * s, 0.70), 0.04, 6, dark)
    # transoms
    box(m, (-0.30, -0.95, 0.55), (0.30, 0.95, 0.80), frame)
    if motor:
        for ax in (-hb, hb):
            d = 1 if ax < 0 else -1
            box(m, (ax + d * 0.30 - 0.32, -0.45, 0.25), (ax + d * 0.30 + 0.32, 0.30, 0.72), dark)
            box(m, (ax - 0.25, 0.30, 0.20), (ax + 0.25, 0.55, 0.62), dark)
    else:
        for ax in (-hb, hb):
            d = 1 if ax < 0 else -1
            box(m, (ax + d * 0.45 - 0.08, -0.6, 0.55), (ax + d * 0.45 + 0.08, 0.6, 0.66), frame)
    if lod == 0:
        # cables
        cylinder(m, (-0.3, 0.6, 0.86), (-1.0, 0.75, 0.62), 0.025, 6, rubber)
        cylinder(m, (0.3, -0.6, 0.86), (1.0, -0.75, 0.62), 0.025, 6, rubber)
    return m


def pantograph(sw, lod):
    """Single-arm pantograph, raised, base at z = 0, pointing to +x."""
    m = Mesh(1)
    seg = 8 if lod == 0 else 5
    ins = sw.uvf('insulator')
    frame = sw.uvf('panto_blue')
    alu = sw.uvf('panto_alu')
    carbon = sw.uvf('panto_carbon')
    for x in (-0.55, 0.55):
        for y in (-0.42, 0.42):
            cylinder(m, (x, y, 0), (x, y, 0.22), 0.07, seg, ins)
    box(m, (-0.70, -0.50, 0.22), (0.70, 0.50, 0.28), frame, faces='xXyYZz')
    knee = np.array([-0.95, 0, 0.70])
    cylinder(m, (0.45, -0.25, 0.30), knee + (0, -0.06, 0), 0.05, seg, frame)
    cylinder(m, (0.45, 0.25, 0.30), knee + (0, 0.06, 0), 0.05, seg, frame)
    cylinder(m, (0.30, 0, 0.30), knee, 0.025, 6, frame)
    head = np.array([0.05, 0, 1.30])
    cylinder(m, knee, head, 0.04, seg, frame)
    cylinder(m, (-0.85, 0, 0.66), head + (-0.05, 0, -0.05), 0.015, 5, frame)
    for dx in (-0.14, 0.14):
        box(m, (head[0] + dx - 0.03, -0.80, head[2] + 0.02), (head[0] + dx + 0.03, 0.80, head[2] + 0.07), alu)
        box(m, (head[0] + dx - 0.025, -0.65, head[2] + 0.07), (head[0] + dx + 0.025, 0.65, head[2] + 0.10), carbon)
        for s in (1, -1):
            cylinder(m, (head[0] + dx, 0.80 * s, head[2] + 0.05), (head[0] + dx, 0.98 * s, head[2] - 0.08), 0.015, 5, alu)
    box(m, (head[0] - 0.16, -0.08, head[2] - 0.08), (head[0] + 0.16, 0.08, head[2] + 0.02), frame)
    return m


def gangway(sw, x_body, x_end, lod):
    """Rubber bellows between body end x_body and coupling plane x_end."""
    m = Mesh(1)
    rub = sw.uvf('bellows')
    frame = sw.uvf('bellows_frame')
    n = 5 if lod == 0 else 1
    z0, z1 = rj.FLOOR_MEZZ - 0.18, 3.75
    sgn = 1 if x_end > x_body else -1
    for k in range(n):
        a = x_body + (x_end - x_body) * k / n
        b = x_body + (x_end - x_body) * (k + 1) / n
        w = 0.98 if k % 2 == 0 else 0.93
        lo, hi = min(a, b), max(a, b)
        box(m, (lo, -w, z0), (hi, w, z1), rub, faces='yYZ')
    # outer frame on the body end
    lo, hi = min(x_body, x_body + sgn * 0.06), max(x_body, x_body + sgn * 0.06)
    box(m, (lo, -1.08, z0 - 0.08), (hi, -0.95, z1 + 0.08), frame)
    box(m, (lo, 0.95, z0 - 0.08), (hi, 1.08, z1 + 0.08), frame)
    box(m, (lo, -0.95, z1), (hi, 0.95, z1 + 0.08), frame)
    # end face of the bellows (dark) - closes the gap visually
    face = 'X' if sgn > 0 else 'x'
    box(m, (min(x_end - sgn * 0.01, x_end), -0.93, z0), (max(x_end - sgn * 0.01, x_end), 0.93, z1), rub, faces=face)
    # semi-permanent coupler bar
    box(m, (min(x_body, x_end), -0.12, 0.92), (max(x_body, x_end), 0.12, 1.12), sw.uvf('bogie_dark'))
    return m


def front_coupler(sw, x_tip, lod):
    """Scharfenberg coupler in the nose recess (pointing +x); x_tip is the coupler face."""
    m = Mesh(1)
    dark = sw.uvf('bogie_dark')
    grey = sw.uvf('coupler_box')
    zc = 1.32
    cylinder(m, (12.72, 0, zc), (x_tip - 0.18, 0, zc), 0.08, 12, dark)
    box(m, (x_tip - 0.22, -0.20, zc - 0.17), (x_tip, 0.20, zc + 0.17), dark)
    box(m, (x_tip - 0.40, -0.26, zc + 0.19), (x_tip - 0.10, 0.26, zc + 0.31), grey)
    if lod == 0:
        cylinder(m, (x_tip, 0.08, zc + 0.07), (x_tip + 0.04, 0.08, zc + 0.07), 0.06, 12, sw.uvf('bogie_frame'))
        cylinder(m, (x_tip, -0.08, zc - 0.07), (x_tip + 0.03, -0.08, zc - 0.07), 0.035, 8, sw.uvf('bogie_frame'))
        cylinder(m, (x_tip - 0.30, 0.15, zc + 0.12), (x_tip - 0.30, 0.15, zc + 0.19), 0.03, 8, sw.uvf('panto_alu'))
        cylinder(m, (x_tip - 0.05, 0.12, zc - 0.17), (x_tip + 0.10, 0.18, zc - 0.55), 0.022, 6, sw.uvf('panto_alu'))
    return m


def roof_equipment(sw, spec, lod, with_panto):
    """Roof recess equipment over the lowered roof ends."""
    m = Mesh(1)
    grey = sw.uvf('roof_box')
    grille = sw.uvf('grille')
    zr = rj.ROOF_END
    a, b = spec.low
    ends = []
    if spec.kind == 'mid':
        ends = [(b + 1.0, spec.x_front - 0.5), (spec.x_rear + 0.5, a - 1.0)]
    else:
        ends = [(spec.x_rear + 0.5, a - 1.0), (b + 1.0, 9.9)]
    for k, (x0, x1) in enumerate(ends):
        # AC unit
        box(m, (x0, -0.85, zr - 0.08), (x0 + 2.2, 0.85, zr + 0.16), grey)
        if lod == 0:
            box(m, (x0 + 0.2, -0.6, zr + 0.16), (x0 + 1.0, 0.6, zr + 0.18), grille, faces='Z')
            cylinder(m, (x0 + 1.6, 0, zr + 0.16), (x0 + 1.6, 0, zr + 0.18), 0.28, 16, grille, caps=True)
        # pantograph area / equipment
        if not (with_panto and k == 0):
            box(m, (x0 + 2.4, -0.6, zr - 0.08), (min(x1, x0 + 3.6), 0.6, zr + 0.06), grey)
    return m


def roof_panto_x(spec):
    a, b = spec.low
    return b + 1.0 + 2.9 if spec.kind == 'mid' else -9.6


# ------------------------------------------------------------------------- lights
def light_patch(centre, normal, w, h, uv_centre, seg=10, round_=True, tilt=0.0):
    m = Mesh(1)
    n = np.asarray(normal, float)
    n /= np.linalg.norm(n)
    up = np.array([0, 0, 1.0]) - n * n[2]
    up /= np.linalg.norm(up)
    rt = np.cross(up, n)
    rt, up = math.cos(tilt) * rt + math.sin(tilt) * up, -math.sin(tilt) * rt + math.cos(tilt) * up
    c = np.asarray(centre, float) + 0.012 * n
    pts = []
    for i in range(seg):
        a = 2 * math.pi * i / seg
        if round_:
            pts.append(c + rt * (w / 2) * math.cos(a) + up * (h / 2) * math.sin(a))
        else:
            sx = np.sign(math.cos(a)) * min(1, abs(math.cos(a)) * 1.6)
            sy = np.sign(math.sin(a)) * min(1, abs(math.sin(a)) * 1.6)
            pts.append(c + rt * (w / 2) * sx + up * (h / 2) * sy)
    uv = [uv_centre] * 3
    for i in range(seg):
        m.tri([c, pts[i], pts[(i + 1) % seg]], uv, [n, n, n], 0, 'light')
    return m


# ------------------------------------------------------------------------- interior
class Interior:
    """Seats, floors, stairs, partitions and lights of one car. Returns mesh + seat list."""

    def __init__(self, spec, sw, layout):
        self.spec = spec
        self.sw = sw
        self.layout = layout   # dict with 'cls' per zone: '1', '2', 'mz'
        self.mesh = Mesh(1)
        self.seats = []        # (x, y, z, facing)
        self.lights = []       # (x, z, length)

    def build(self, loft):
        spec, m, sw = self.spec, self.mesh, self.sw
        a, b = spec.low
        wall = sw.uvf('wall')
        floor = sw.uvf('floor')
        ceil = sw.uvf('ceiling')
        # inner wall surface
        xlim = (spec.x_rear + 0.06, (13.3 if spec.kind == 'end' else spec.x_front - 0.06))
        if spec.kind == 'end':
            loft.build_inner(m, 0, wall, (xlim[0], rj.XS))
            loft.build_inner(m, 0, sw.uvf('cab_wall'), (rj.XS, xlim[1]))
        else:
            loft.build_inner(m, 0, wall, xlim)
        # ---- floors
        hw = 1.33
        xr, xf = xlim[0], (11.9 if spec.kind == 'end' else xlim[1])
        box(m, (a + 0.05, -hw, rj.FLOOR_LOW - 0.08), (b - 0.05, hw, rj.FLOOR_LOW), floor, faces='Z')
        # upper deck slab (leaves a stair well at each end on +y side)
        box(m, (a + 1.7, -hw, rj.CEIL_LOW), (b - 1.7, hw, rj.FLOOR_UP), floor, faces='Zz')
        box(m, (a + 0.05, -hw, rj.CEIL_LOW), (a + 1.7, 0.0, rj.FLOOR_UP), floor, faces='Zzx')
        box(m, (b - 1.7, -hw, rj.CEIL_LOW), (b - 0.05, 0.0, rj.FLOOR_UP), floor, faces='ZzX')
        # mezzanine floors + their walls down to the lower deck
        box(m, (xr, -hw, rj.FLOOR_MEZZ - 0.1), (a + 0.05, hw, rj.FLOOR_MEZZ), floor, faces='ZX')
        box(m, (b - 0.05, -hw, rj.FLOOR_MEZZ - 0.1), (xf, hw, rj.FLOOR_MEZZ), floor, faces='Zx')
        # mezzanine ceilings
        box(m, (xr, -hw, 3.75), (a + 0.05, hw, 3.80), ceil, faces='z')
        box(m, (b - 0.05, -hw, 3.75), (min(xf, 11.3), hw, 3.80), ceil, faces='z')
        # stairs: down to the lower deck (-y half), up to the upper deck (+y half)
        for edge, d in ((a, 1), (b, -1)):
            self._stairs(edge, d, -1.0, -0.15, rj.FLOOR_MEZZ, rj.FLOOR_LOW, 0.9)
            self._stairs(edge, d, 0.15, 1.0, rj.FLOOR_MEZZ, rj.FLOOR_UP, 1.65)
            # glass/white partition between the stairs
            box(m, (min(edge, edge + d * 1.65), -0.15, rj.FLOOR_MEZZ), (max(edge, edge + d * 1.65), -0.12, 2.5), sw.uvf('partition'))
        # bulkheads at the gangway ends with a door
        for x, d in ((xr, 1), (xf if spec.kind == 'mid' else 10.0, -1)):
            face = 'X' if d > 0 else 'x'
            box(m, (x - 0.01, -hw, rj.FLOOR_MEZZ), (x + 0.01, hw, 4.2), wall, faces=face)
        # ---- seats
        cls = self.layout
        for zone, (x0, x1, zf) in self._zones().items():
            kind = cls.get(zone, '2')
            self._seat_rows(x0, x1, zf, kind, zone)
        # luggage racks over the mezzanine seats
        for x0, x1 in ((xr + 0.3, a - 0.4), (b + 0.4, (xf - 0.3) if spec.kind == 'mid' else 9.6)):
            for s in (1, -1):
                box(m, (x0, 0.98 * s - 0.20, 3.20), (x1, 0.98 * s + 0.20, 3.24), sw.uvf('rack'))
        # cab
        if spec.kind == 'end':
            self._cab()
        # lights (CYLINDRICAL) per deck
        self.lights = [((a + b) / 2, rj.CEIL_LOW - 0.05, (b - a) - 1),
                       ((a + b) / 2, 4.30, (b - a) - 1),
                       ((xr + a) / 2, 3.70, a - xr - 0.5),
                       ((b + xf) / 2, 3.70, xf - b - 0.5)]
        return self

    def _zones(self):
        spec = self.spec
        a, b = self.spec.low
        xr = spec.x_rear + 0.15
        xf = spec.x_front - 0.15 if spec.kind == 'mid' else 9.8
        zones = {
            'lower': (a + 1.15, b - 1.15, rj.FLOOR_LOW),
            'upper': (a + 1.80, b - 1.80, rj.FLOOR_UP),
            'mezz_r': (xr + 0.1, a - 0.55, rj.FLOOR_MEZZ),
            'mezz_f': (b + 0.55, xf, rj.FLOOR_MEZZ),
        }
        return zones

    def _stairs(self, edge, d, y0, y1, z0, z1, length):
        m = self.mesh
        n = max(2, int(round(abs(z1 - z0) / 0.19)))
        step = length / n
        for k in range(n):
            xa = edge + d * step * k
            xb = edge + d * step * (k + 1)
            z = z0 + (z1 - z0) * (k + 1) / n
            zl = min(z, z0 + (z1 - z0) * k / n)
            box(m, (min(xa, xb), y0, min(zl, z) - 0.02), (max(xa, xb), y1, max(zl, z)), self.sw.uvf('step'), faces='ZxX')

    def _seat(self, x, y, zf, facing, kind):
        m, sw = self.mesh, self.sw
        cush = sw.uvf('seat1' if kind == '1' else 'seat2')
        head = sw.uvf('seat1_head' if kind == '1' else 'seat2_head')
        w = 0.50 if kind == '1' else 0.46
        d = facing
        box(m, (x - 0.24, y - w / 2, zf + 0.40), (x + 0.24, y + w / 2, zf + 0.50), cush)
        bx = x - d * 0.24
        box(m, (min(bx, bx - d * 0.10), y - w / 2, zf + 0.45), (max(bx, bx - d * 0.10), y + w / 2, zf + 1.02), cush)
        box(m, (min(bx, bx - d * 0.11), y - w / 2 + 0.02, zf + 1.02), (max(bx, bx - d * 0.11), y + w / 2 - 0.02, zf + 1.22), head)
        box(m, (x - 0.12, y - 0.10, zf), (x + 0.05, y + 0.10, zf + 0.40), sw.uvf('seat_frame'), faces='xXyY')
        self.seats.append((x, y, zf, d))

    def _seat_rows(self, x0, x1, zf, kind, zone):
        if kind == 'mz':
            # multi-purpose area: tip-up seats on one side, bikes / wheelchairs on the other
            n = int((x1 - x0) // 0.9)
            for k in range(n):
                x = x0 + 0.45 + k * 0.9
                self._seat(x, 1.0, zf, 1, '2')
            for k in range(int((x1 - x0) // 0.7)):
                xx = x0 + 0.35 + k * 0.7
                cylinder(self.mesh, (xx, -1.15, zf + 0.35), (xx, -0.75, zf + 0.35), 0.33, 10, self.sw.uvf('bike'), caps=False, smooth=False)
            return
        pitch = 1.00 if kind == '1' else 0.90
        ys = (-0.95, 0.75, 1.0) if kind == '1' else (-1.0, -0.55, 0.55, 1.0)
        if kind == '1':
            ys = (-0.98, 0.70, 1.0)
            ys = (-0.95, 0.55, 1.02)
        upper = zone == 'upper'
        if upper:
            ys = tuple(np.clip(y, -0.9, 0.9) for y in ys)
        n = int((x1 - x0) // pitch)
        if n <= 0:
            return
        start = x0 + ((x1 - x0) - n * pitch) / 2 + pitch / 2
        for k in range(n):
            x = start + k * pitch
            facing = 1 if (k // 2) % 2 == 0 else -1
            if k % 2 == 1:
                facing = -facing
            facing = 1 if k % 2 == 0 else -1
            for y in ys:
                self._seat(x, y, zf, facing, kind)
            if facing == 1 and k + 1 < n:
                # table between face-to-face seats
                xt = x + pitch / 2
                for s in (-1, 1):
                    yy = 0.95 * s if kind == '2' else (0.95 * s if s < 0 else 0.78)
                    box(self.mesh, (xt - 0.20, yy - 0.25, zf + 0.70), (xt + 0.20, yy + 0.25, zf + 0.74), self.sw.uvf('table'))

    def _cab(self):
        m, sw = self.mesh, self.sw
        zf = rj.FLOOR_MEZZ
        box(m, (10.0, -1.25, zf), (10.05, 1.25, 3.9), sw.uvf('cab_wall'), faces='xX')
        box(m, (11.9, -0.95, zf - 0.05), (12.75, 0.95, zf), sw.uvf('cab_floor'), faces='Z')
        box(m, (11.75, -0.95, zf), (12.45, 0.95, zf + 0.90), sw.uvf('desk'))
        box(m, (11.85, -0.55, zf + 0.90), (12.05, 0.55, zf + 1.15), sw.uvf('screen'), faces='xXyYZ')
        box(m, (11.0, -0.28, zf + 0.42), (11.45, 0.28, zf + 0.55), sw.uvf('seat_cab'))
        box(m, (10.85, -0.28, zf + 0.55), (11.0, 0.28, zf + 1.30), sw.uvf('seat_cab'))

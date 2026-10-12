"""Triangle mesh container and Transport Fever 3 .msh / .msh.blob writer.

Coordinate system (as in the game): X along the vehicle (front = +X), Y to the left,
Z up, origin on the rail top at the vehicle centre. Units are metres.

The blob stores float32 attribute arrays (position xyz, uv0 uv, normal xyz, tangent xyzw)
followed by one uint32 index stream per attribute and sub-mesh. Offsets and counts in the
.msh file are given in bytes. Texture v = 0 is the top row of the DDS image.
"""
import math
import numpy as np


class Mesh:
    def __init__(self, nslots=1):
        self.slots = [dict(p=[], uv=[], n=[], paint=[]) for _ in range(nslots)]

    # ------------------------------------------------------------------ building
    def tri(self, p, uv, n=None, slot=0, paint=None):
        """p: 3 points, uv: 3 uvs, n: optional 3 normals (else flat)."""
        p = np.asarray(p, np.float64)
        e1, e2 = p[1] - p[0], p[2] - p[0]
        fn = np.cross(e1, e2)
        area = np.linalg.norm(fn)
        if area < 1e-9:
            return
        if n is None:
            n = np.repeat((fn / area)[None], 3, 0)
        s = self.slots[slot]
        s['p'].append(p)
        s['uv'].append(np.asarray(uv, np.float64))
        s['n'].append(np.asarray(n, np.float64))
        s['paint'].append(paint)

    def quad(self, p, uv, n=None, slot=0, paint=None):
        """Quad p0..p3 counter-clockwise seen from the outside."""
        if n is None:
            self.tri([p[0], p[1], p[2]], [uv[0], uv[1], uv[2]], None, slot, paint)
            self.tri([p[0], p[2], p[3]], [uv[0], uv[2], uv[3]], None, slot, paint)
        else:
            self.tri([p[0], p[1], p[2]], [uv[0], uv[1], uv[2]], [n[0], n[1], n[2]], slot, paint)
            self.tri([p[0], p[2], p[3]], [uv[0], uv[2], uv[3]], [n[0], n[2], n[3]], slot, paint)

    def extend(self, other, M=None, slot_map=None):
        for si, s in enumerate(other.slots):
            ds = self.slots[slot_map[si] if slot_map else si]
            for p, uv, n, paint in zip(s['p'], s['uv'], s['n'], s['paint']):
                if M is not None:
                    p = p @ M[:3, :3].T + M[:3, 3]
                    n = n @ M[:3, :3].T
                    if np.linalg.det(M[:3, :3]) < 0:
                        p = p[::-1]
                        n = n[::-1]
                        uv = uv[::-1]
                ds['p'].append(p)
                ds['uv'].append(uv)
                ds['n'].append(n)
                ds['paint'].append(paint)

    def transformed(self, M):
        m = Mesh(len(self.slots))
        m.extend(self, M)
        return m

    def ntris(self):
        return sum(len(s['p']) for s in self.slots)

    def arrays(self, slot=None):
        """Concatenated (T,3,3) positions, (T,3,2) uvs, (T,3,3) normals over slots."""
        slots = self.slots if slot is None else [self.slots[slot]]
        P = [np.array(s['p']).reshape(-1, 3, 3) for s in slots if s['p']]
        U = [np.array(s['uv']).reshape(-1, 3, 2) for s in slots if s['p']]
        N = [np.array(s['n']).reshape(-1, 3, 3) for s in slots if s['p']]
        if not P:
            return np.zeros((0, 3, 3)), np.zeros((0, 3, 2)), np.zeros((0, 3, 3))
        return np.concatenate(P), np.concatenate(U), np.concatenate(N)

    def bbox(self):
        P = self.arrays()[0].reshape(-1, 3)
        return P.min(0), P.max(0)

    # ------------------------------------------------------------------ export
    def write(self, path_msh):
        attrs = dict(position=[], uv0=[], normal=[], tangent=[])
        idx_streams = []
        base = dict(position=0, uv0=0, normal=0, tangent=0)
        for s in self.slots:
            if not s['p']:
                idx_streams.append(None)
                continue
            P = np.array(s['p']).reshape(-1, 3, 3)
            U = np.array(s['uv']).reshape(-1, 3, 2)
            N = np.array(s['n']).reshape(-1, 3, 3)
            N /= np.linalg.norm(N, axis=2, keepdims=True) + 1e-12
            T = _tangents(P, U, N)
            streams = {}
            for name, arr in (('position', P.reshape(-1, 3)), ('uv0', U.reshape(-1, 2)),
                              ('normal', N.reshape(-1, 3)), ('tangent', T.reshape(-1, 4))):
                q = np.round(arr, 5)
                uniq, inv = np.unique(q, axis=0, return_inverse=True)
                attrs[name].append(uniq.astype(np.float32))
                streams[name] = (inv.reshape(-1) + base[name]).astype(np.uint32)
                base[name] += len(uniq)
            idx_streams.append(streams)
        blob = b''
        meta_attr = {}
        for name, comp in (('position', 3), ('uv0', 2), ('normal', 3), ('tangent', 4)):
            data = np.concatenate(attrs[name]).astype('<f4').tobytes() if attrs[name] else b''
            meta_attr[name] = (len(data), comp, len(blob))
            blob += data
        meta_sub = []
        for st in idx_streams:
            if st is None:
                continue
            entry = {}
            for name in ('position', 'uv0', 'normal', 'tangent'):
                data = st[name].astype('<u4').tobytes()
                entry[name] = (len(data), len(blob))
                blob += data
            meta_sub.append(entry)
        with open(path_msh + '.blob', 'wb') as f:
            f.write(blob)
        out = ['function data()', 'return { ', '\t\tsubMeshes = {']
        for e in meta_sub:
            out.append('\t\t\t{ ')
            out.append('\t\t\t\tindices = { ')
            for name in ('normal', 'position', 'tangent', 'uv0'):
                c, o = e[name]
                out += ['\t\t\t\t\t%s = { ' % name, '\t\t\t\t\t\tcount = %d,' % c,
                        '\t\t\t\t\t\toffset = %d,' % o, '\t\t\t\t\t},']
            out += ['\t\t\t\t},', '\t\t\t},']
        out += ['\t\t},', '\t\tvertexAttr = { ']
        for name in ('normal', 'position', 'tangent', 'uv0'):
            c, comp, o = meta_attr[name]
            out += ['\t\t\t%s = { ' % name, '\t\t\t\tcount = %d,' % c, '\t\t\t\tnumComp = %d,' % comp,
                    '\t\t\t\toffset = %d,' % o, '\t\t\t},']
        out += ['\t\t},', '\t}', 'end']
        with open(path_msh, 'w', newline='\n') as f:
            f.write('\n'.join(out))
        return sum(1 for s in idx_streams if s is not None)

    def used_slots(self):
        return [i for i, s in enumerate(self.slots) if s['p']]


def _tangents(P, U, N):
    e1 = P[:, 1] - P[:, 0]
    e2 = P[:, 2] - P[:, 0]
    d1 = U[:, 1] - U[:, 0]
    d2 = U[:, 2] - U[:, 0]
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    ok = np.abs(det) > 1e-12
    r = np.where(ok, 1.0 / np.where(ok, det, 1), 0)
    t = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    b = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    fallback = np.where(np.abs(e1).sum(1, keepdims=True) > 0, e1, np.array([[1.0, 0, 0]]))
    t = np.where(ok[:, None], t, fallback)
    T = np.repeat(t[:, None], 3, 1)
    B = np.repeat(b[:, None], 3, 1)
    T = T - N * (N * T).sum(2, keepdims=True)
    bad = np.linalg.norm(T, axis=2) < 1e-9
    alt = np.cross(N, np.array([0, 0, 1.0]))
    alt2 = np.cross(N, np.array([0, 1.0, 0]))
    alt = np.where((np.linalg.norm(alt, axis=2) < 1e-6)[..., None], alt2, alt)
    T = np.where(bad[..., None], alt, T)
    T /= np.linalg.norm(T, axis=2, keepdims=True) + 1e-12
    w = np.where((np.cross(N, T) * B).sum(2) < 0, -1.0, 1.0)
    return np.concatenate([T, w[..., None]], 2)


# ---------------------------------------------------------------------- matrices
def translate(x, y, z):
    M = np.eye(4)
    M[:3, 3] = (x, y, z)
    return M


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    M[0, 0], M[0, 1], M[1, 0], M[1, 1] = c, -s, s, c
    return M


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    M[0, 0], M[0, 2], M[2, 0], M[2, 2] = c, s, -s, c
    return M


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    M = np.eye(4)
    M[1, 1], M[1, 2], M[2, 1], M[2, 2] = c, -s, s, c
    return M


def scale(x, y, z):
    return np.diag([x, y, z, 1.0])


def lua_transf(M):
    """Column-major 16-number list as used in .mdl files."""
    return '{ ' + ', '.join(_fmt(v) for v in M.T.reshape(-1)) + ', }'


def _fmt(v):
    v = float(v)
    if abs(v) < 5e-7:
        v = 0.0
    s = ('%.5f' % v).rstrip('0').rstrip('.')
    return '0' if s in ('-0', '') else s


# ---------------------------------------------------------------------- primitives
def box(mesh, lo, hi, uvf, slot=0, paint=None, faces='xXyYzZ'):
    """Axis aligned box. uvf(face, corner_points) -> 4 uvs."""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    F = {
        'X': [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
        'x': [(x0, y1, z0), (x0, y0, z0), (x0, y0, z1), (x0, y1, z1)],
        'Y': [(x1, y1, z0), (x0, y1, z0), (x0, y1, z1), (x1, y1, z1)],
        'y': [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        'Z': [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        'z': [(x0, y1, z0), (x1, y1, z0), (x1, y0, z0), (x0, y0, z0)],
    }
    for f in faces:
        pts = F[f]
        mesh.quad(pts, uvf(f, pts), None, slot, paint)


def cylinder(mesh, p0, p1, r, seg, uvf, slot=0, paint=None, caps=True, r1=None, smooth=True):
    """Cylinder (or cone frustum) from p0 to p1."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    r1 = r if r1 is None else r1
    ax = p1 - p0
    L = np.linalg.norm(ax)
    ax /= L
    ref = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(ax, ref)
    u /= np.linalg.norm(u)
    v = np.cross(ax, u)
    ring = [(math.cos(2 * math.pi * i / seg), math.sin(2 * math.pi * i / seg)) for i in range(seg + 1)]
    for i in range(seg):
        c0, s0 = ring[i]
        c1, s1 = ring[i + 1]
        d0 = c0 * u + s0 * v
        d1 = c1 * u + s1 * v
        pts = [p0 + r * d0, p0 + r * d1, p1 + r1 * d1, p1 + r1 * d0]
        n = [d0, d1, d1, d0] if smooth else None
        mesh.quad(pts, uvf('side', (i / seg, (i + 1) / seg)), n, slot, paint)
    if caps:
        for cen, rr, sign in ((p0, r, -1), (p1, r1, 1)):
            for i in range(seg):
                c0, s0 = ring[i]
                c1, s1 = ring[i + 1]
                a = cen + rr * (c0 * u + s0 * v)
                b = cen + rr * (c1 * u + s1 * v)
                tri = [cen, a, b] if sign > 0 else [cen, b, a]
                mesh.tri(tri, uvf('cap', (c0, s0, c1, s1)), None, slot, paint)


def disc_ring(mesh, centre, axis, r_in, r_out, seg, uvf, slot=0, paint=None):
    """Flat annulus facing +axis."""
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(axis, ref)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    c = np.asarray(centre, float)
    for i in range(seg):
        a0 = 2 * math.pi * i / seg
        a1 = 2 * math.pi * (i + 1) / seg
        d0 = math.cos(a0) * u + math.sin(a0) * v
        d1 = math.cos(a1) * u + math.sin(a1) * v
        pts = [c + r_in * d0, c + r_out * d0, c + r_out * d1, c + r_in * d1]
        mesh.quad(pts, uvf('ring', (i, seg)), None, slot, paint)

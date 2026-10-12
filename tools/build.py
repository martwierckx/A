"""Build the Transport Fever 3 mod "OeBB Railjet Doppelstock" into ../mod/<MOD_ID>.

Usage: python3 tools/build.py [--quick]
  --quick  skips texture compression of the big atlases (for geometry checks).
"""
import json
import math
import os
import shutil
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import dds
import paint
import parts
import railjet as rj
from meshlib import Mesh, translate, rot_z, lua_transf, box
from render import Renderer

MOD_ID = 'claude_railjet_dosto'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mod', MOD_ID))
VEH = os.path.join(ROOT, 'content', 'vehicle', 'train', 'rj_dosto')
ATLAS = 4096
INTERIOR_PX = 1024
QUICK = '--quick' in sys.argv

CHARTS = {'end1': (0.0, 0.25), 'mid': (0.25, 0.5), 'end2': (0.5, 0.75)}
GLASS_REGIONS = {'window': (0.0, 0.0, 0.5, 1.0), 'windscreen': (0.5, 0.0, 1.0, 1.0)}
LIGHT_WHITE = (0.25, 0.25)
LIGHT_RED = (0.25, 0.75)

# headlight clusters on the nose: (y, z, radius) - lower LED clusters, top light
HEADLIGHTS = [(0.98, 2.32, 0.10), (-0.98, 2.32, 0.10)]
TOPLIGHT = (0.0, 4.32, 0.09)
TAILLIGHTS = [(0.74, 2.33, 0.05), (-0.74, 2.33, 0.05)]

ROT180 = rot_z(math.pi)

VARIANTS = [
    # name, spec kind, chart, class text, cab at +x?, layout, capacity, powered, panto
    dict(name='rj_dosto_1_front', kind='end', chart='end1', cls='1', cab_front=True,
         layout={'lower': '1', 'upper': '1', 'mezz_r': '1', 'mezz_f': '1'}, capacity=58, powered=True, panto=False),
    dict(name='rj_dosto_2_mid', kind='mid', chart='mid', cls='2',
         layout={'lower': '2', 'upper': '2', 'mezz_r': '2', 'mezz_f': '2'}, capacity=96, powered=False, panto=True),
    dict(name='rj_dosto_3_mz', kind='mid', chart='mid', cls='2',
         layout={'lower': 'mz', 'upper': '2', 'mezz_r': '2', 'mezz_f': '2'}, capacity=66, powered=False, panto=False),
    dict(name='rj_dosto_4_mid', kind='mid', chart='mid', cls='2',
         layout={'lower': '2', 'upper': '2', 'mezz_r': '2', 'mezz_f': '2'}, capacity=96, powered=False, panto=False),
    dict(name='rj_dosto_6_rear', kind='end', chart='end2', cls='2', cab_front=False,
         layout={'lower': '2', 'upper': '2', 'mezz_r': '2', 'mezz_f': '2'}, capacity=74, powered=True, panto=False),
]

EMPTY_KG = {'end': 56000, 'mid': 47000}
POWER_KW = 3000          # per driving trailer (two motor bogies) - estimate
TRACTIVE_KN = 150        # per driving trailer - estimate
TOP_SPEED = 200 / 3.6


def ensure(p):
    os.makedirs(p, exist_ok=True)
    return p


# ============================================================================ geometry
class CarGeometry:
    def __init__(self, var, bsw, isw):
        self.var = var
        self.spec = rj.CarSpec(var['kind'])
        self.bsw = bsw
        self.isw = isw
        chart = CHARTS[var['chart']]
        self.lofts = {lod: rj.Loft(self.spec, lod, chart) for lod in (0, 1, 2)}

    def body(self, lod):
        spec, sw = self.spec, self.bsw
        lf = self.lofts[lod]
        m = Mesh(1)
        lf.build_shell(m, 0)
        if lod == 0:
            lf.build_reveals(m, 0, sw.uvf('gasket'), None, None)
        # end walls + gangways
        rear_body = spec.x_rear
        lf.end_cap(m, 0, rear_body, -1, sw.uvf('endwall'))
        m.extend(parts.gangway(sw, rear_body, -spec.half_len, lod))
        if spec.kind == 'mid':
            lf.end_cap(m, 0, spec.x_front, 1, sw.uvf('endwall'))
            m.extend(parts.gangway(sw, spec.x_front, spec.half_len, lod))
        else:
            m.extend(parts.front_coupler(sw, spec.half_len - 0.05, lod))
            if lod == 0:
                self._front_details(m)
        # roof equipment / pantograph
        m.extend(parts.roof_equipment(sw, spec, lod, self.var['panto']))
        if self.var['panto']:
            px = parts.roof_panto_x(spec)
            m.extend(parts.pantograph(sw, lod), translate(px, 0, rj.ROOF_END - 0.06))
        # bogie frames (static)
        for k, bx in enumerate(spec.bogies):
            motor = self.var['powered']
            m.extend(parts.bogie(sw, motor, lod), translate(bx, 0, 0))
        # underbody cover between the bogies
        a, b = spec.low
        box(m, (a + 0.1, -1.25, rj.ZB_LOW - 0.02), (b - 0.1, 1.25, rj.ZB_LOW), sw.uvf('bogie_dark'), faces='z')
        return m

    def _front_details(self, m):
        sw = self.bsw
        lf = self.lofts[0]
        for y0, y1 in ((0.10, 0.80), (-0.62, 0.08)):
            h0 = lf.surface_point(y0, 2.60)
            h1 = lf.surface_point(y1, 3.15)
            if h0 and h1:
                p0 = h0[0] + h0[1] * 0.03
                p1 = h1[0] + h1[1] * 0.03
                parts.cylinder(m, p0, p1, 0.013, 6, sw.uvf('bogie_dark'))

    def axles(self):
        out = []
        for bx in self.spec.bogies:
            for d in (-1, 1):
                out.append(bx + d * parts.WHEELBASE / 2)
        return out

    def interior(self):
        it = parts.Interior(self.spec, self.isw, self.var['layout']).build(self.lofts[0])
        return it

    def glass(self):
        m = Mesh(1)
        self.lofts[0].glass(m, 0, GLASS_REGIONS)
        return m

    def doors(self, lod):
        """List of (mesh, side, leaf, door_idx)."""
        out = []
        lf = self.lofts[lod]
        for di in range(len(self.spec.doors)):
            for side in (1, -1):
                for leaf in (-1, 1):
                    mesh, _ = lf.door_leaves(di, side, leaf)
                    out.append((mesh, side, leaf, di))
        return out

    def light_points(self):
        """3D centres and radii of the light bezels at the +x cab (for the texture bake)."""
        out = []
        lf = self.lofts[0]
        for (y, z, r) in HEADLIGHTS + TAILLIGHTS:
            hit = lf.surface_point(y, z)
            if hit:
                out.append((hit[0], 0.42 if (y, z, r) in HEADLIGHTS else 0.11))
        return out

    def lights(self):
        """Emissive meshes at the +x cab: (white, red)."""
        lf = self.lofts[0]
        white = Mesh(1)
        red = Mesh(1)
        self.light_pos = []
        for (y, z, r) in HEADLIGHTS:
            hit = lf.surface_point(y, z)
            p, n = hit
            white.extend(parts.light_patch(p, n, 0.58, 0.14, LIGHT_WHITE, round_=False,
                                           tilt=math.copysign(0.38, y)))
            self.light_pos.append(('head', p, n))
        hit = lf.surface_point(TOPLIGHT[0], TOPLIGHT[1])
        if hit:
            p, n = hit
            white.extend(parts.light_patch(p, n, 0.30, 0.10, LIGHT_WHITE, round_=False))
            self.light_pos.append(('top', p, n))
        for (y, z, r) in TAILLIGHTS:
            hit = lf.surface_point(y, z)
            p, n = hit
            red.extend(parts.light_patch(p, n, 0.14, 0.08, LIGHT_RED, round_=False))
            self.light_pos.append(('tail', p, n))
        return white, red


# ============================================================================ textures
def build_textures(geoms):
    tex_dir = ensure(os.path.join(VEH, 'mat', 'tex'))
    bsw = geoms[0].bsw
    alb = np.zeros((ATLAS, ATLAS, 3))
    mga = np.zeros((ATLAS, ATLAS, 3))
    hgt = np.zeros((ATLAS, ATLAS))
    drt = np.zeros((ATLAS, ATLAS))
    done = set()
    lights = geoms[0].light_points()
    for g in geoms:
        ch = g.var['chart']
        if ch in done:
            continue
        done.add(ch)
        paint.bake_loft_chart(g.spec, CHARTS[ch], alb, mga, hgt, drt, ATLAS, g.var['cls'], lights)
    bsw.paint(alb, mga)
    valid = np.zeros((ATLAS, ATLAS), bool)
    valid[:int(0.75 * ATLAS)] = True
    valid[int(0.875 * ATLAS):] = True
    alb = paint.dilate(alb, valid, 4)
    nrm = paint.normal_from_height(hgt, 1.5)
    cbl = np.stack([np.full(drt.shape, 255.0), drt, np.full(drt.shape, 12.0)], -1)
    small = lambda a, f: a.reshape(a.shape[0] // f, f, a.shape[1] // f, f, -1).mean((1, 3))
    images = {
        'body_albedo': (alb, 'DXT1'),
        'body_metal_gloss_ao': (small(mga, 2), 'DXT1'),
        'body_normal': (small(nrm, 2), 'ATI2'),
        'body_cblend_dirt_rust': (small(cbl, 4), 'DXT1'),
    }
    irgb, imga = paint.interior_texture(geoms[0].isw, INTERIOR_PX)
    images['interior_albedo'] = (irgb, 'DXT1')
    images['interior_metal_gloss_ao'] = (imga, 'DXT1')
    glass = paint.glass_texture(512)
    images['glass_albedo_opacity'] = (glass, 'DXT5')
    images['flat_normal'] = (np.full((64, 64, 2), 128.0), 'ATI2')
    images['glass_metal_gloss_ao'] = (np.broadcast_to(np.array([0, 242, 255.0]), (64, 64, 3)), 'DXT1')
    for name, (img, fmt) in images.items():
        if QUICK and img.shape[0] > 1024:
            img = img[::4, ::4]
        dds.save(os.path.join(tex_dir, name + '.dds'), img, fmt)
    return alb / 255, mga / 255, irgb / 255, glass


# ============================================================================ materials
def write_materials():
    mat = ensure(os.path.join(VEH, 'mat'))
    shared = '''			map_dirt = {
				fragmentSamplers = {
					dirtTex = {
						fileName = "::/vehicle/shared/mat/tex/dirt_albedo.dds",
						type = "TWOD",
					},
				},
			},
			map_dirt_normal = {
				fragmentSamplers = {
					dirtNormalTex = {
						fileName = "::/vehicle/shared/mat/tex/dirt_normal.dds",
						redGreen = true,
						type = "TWOD",
					},
				},
			},
			map_rust = {
				fragmentSamplers = {
					rustTex = {
						fileName = "::/vehicle/shared/mat/tex/rust_albedo.dds",
						type = "TWOD",
					},
				},
			},
			map_rust_normal = {
				fragmentSamplers = {
					rustNormalTex = {
						fileName = "::/vehicle/shared/mat/tex/rust_normal.dds",
						redGreen = true,
						type = "TWOD",
					},
				},
			},
'''

    def physical(prefix, dirt, normal):
        s = ['function data()', 'return {', '\t\t__version = "",', '\t\torder = 0,', '\t\tparams = {']
        s.append('''			color_blend = {
				fragmentProperties = {
					{
						albedoScales = { 0, 0, },
						colors = {
							{ -1, -1, -1, },
							{ -1, -1, -1, },
						},
					},
				},
			},
			dirt_rust = {
				fragmentProperties = {
					{
						dirtColor = { 0.2157, 0.1569, 0.1137, },
						dirtOpacity = %s,
						dirtScale = 8,
						rustColor = { 0.2431, 0.1725, 0.1412, },
						rustOpacity = 0.3,
						rustScale = 2.876,
					},
				},
			},
			light_receiver = {
				fragmentProperties = {
					{
						isLegacyMaterial = false,
						lightMask = 2,
					},
				},
			},
			map_albedo = {
				fragmentSamplers = {
					albedoTex = {
						fileName = "tex/%s_albedo.dds",
						type = "TWOD",
					},
				},
			},
			map_cblend_dirt_rust = {
				fragmentSamplers = {
					cblendDirtRustTex = {
						fileName = "tex/%s",
						type = "TWOD",
					},
				},
			},
''' % (dirt, prefix, 'body_cblend_dirt_rust.dds'))
        s.append(shared)
        s.append('''			map_metal_gloss_ao = {
				fragmentSamplers = {
					metalGlossAoTex = {
						fileName = "tex/%s_metal_gloss_ao.dds",
						type = "TWOD",
					},
				},
			},
			map_normal = {
				fragmentSamplers = {
					normalTex = {
						fileName = "%s",
						redGreen = true,
						type = "TWOD",
					},
				},
			},
''' % (prefix, normal))
        s += ['\t\t},', '\t\ttype = "PHYSICAL_NRML_MAP_CBLEND_DIRT",', '\t}', 'end', '']
        return '\n'.join(s)

    with open(os.path.join(mat, 'body.mtl'), 'w', newline='\n') as f:
        f.write(physical('body', '0.1', 'tex/body_normal.dds'))
    with open(os.path.join(mat, 'interior.mtl'), 'w', newline='\n') as f:
        f.write(physical('interior', '0.0', 'tex/flat_normal.dds'))
    for name, order in (('glass', 3),):
        txt = '''function data()
return {
		__version = "",
		order = %d,
		params = {
			alpha_test = {
				fragmentProperties = {
					{
						cutout = false,
						sorted = false,
					},
				},
			},
			light_receiver = {
				fragmentProperties = {
					{
						isLegacyMaterial = false,
						lightMask = 2,
					},
				},
			},
			map_albedo_opacity = {
				fragmentSamplers = {
					albedoOpacityTex = {
						fileName = "tex/glass_albedo_opacity.dds",
						type = "TWOD",
					},
				},
			},
			map_metal_gloss_ao = {
				fragmentSamplers = {
					metalGlossAoTex = {
						fileName = "tex/glass_metal_gloss_ao.dds",
						type = "TWOD",
					},
				},
			},
		},
		type = "PHYS_TRANSPARENT",
	}
end
''' % order
        with open(os.path.join(mat, name + '.mtl'), 'w', newline='\n') as f:
            f.write(txt)


# ============================================================================ animations
def write_door_anis():
    d = ensure(os.path.join(VEH, 'ani', 'doors'))
    n = 49
    times = [i * 2000 / (n - 1) for i in range(n)]
    for side in (1, -1):
        for leaf in (-1, 1):
            tr = []
            for t in times:
                f = t / 2000
                out = min(1, f / 0.2)
                out = out * out * (3 - 2 * out)
                sl = min(1, max(0, (f - 0.15) / 0.85))
                sl = sl * sl * (3 - 2 * sl)
                M = translate(leaf * 0.64 * sl, side * 0.055 * out, 0)
                tr.append(lua_transf(M))
            name = 'door_%s_%s.ani' % ('l' if side > 0 else 'r', 'rear' if leaf < 0 else 'front')
            with open(os.path.join(d, name), 'w', newline='\n') as f:
                f.write('function data()\nreturn { \n\t\ttimes = { ' + ', '.join('%g' % t for t in times) + ', },\n')
                f.write('\t\ttransfs = {\n' + ''.join('\t\t\t%s,\n' % t for t in tr) + '\t\t},\n\t}\nend\n')


# ============================================================================ mdl
def ani_block(entries, indent):
    t = '\t' * indent
    out = [t + 'animations = { ']
    for key, aid, extra in entries:
        out.append(t + '\t%s = { ' % key)
        for e in extra:
            out.append(t + '\t\t' + e)
        out += [t + '\t\tparams = { ', t + '\t\t\tid = "%s",' % aid, t + '\t\t},', t + '\t\ttype = "FILE_REF",', t + '\t},']
    out.append(t + '},')
    return out


def node(mesh, mats, name, M=None, anims=None, indent=6):
    t = '\t' * indent
    out = [t + '{ ']
    if anims:
        out += ani_block(anims, indent + 1)
    out += [t + '\tmaterials = { %s },' % ''.join('"%s", ' % m for m in mats),
            t + '\tmesh = "%s",' % mesh,
            t + '\tname = "%s",' % name,
            t + '\ttransf = %s,' % lua_transf(M if M is not None else np.eye(4)),
            t + '},']
    return out


def door_anims(side, leaf):
    aid = 'ani/doors/door_%s_%s.ani' % ('l' if side > 0 else 'r', 'rear' if leaf < 0 else 'front')
    sk = 'left' if side > 0 else 'right'
    return [('close_all_doors', aid, ['forward = false,']), ('close_doors_%s' % sk, aid, ['forward = false,']),
            ('open_all_doors', aid, []), ('open_doors_%s' % sk, aid, [])]


def light_anims(kind):
    base = '::/vehicle/shared/ani/%s_parts_%s.ani'
    return [('%s_parts_off' % kind, base % (kind, 'off'), ['interpolation = "NEAREST",']),
            ('%s_parts_on' % kind, base % (kind, 'on'), ['interpolation = "NEAREST",'])]


def export_car(g, rot, msh_dir, prefix):
    """Write all meshes for one variant; returns info for the mdl."""
    var = g.var
    R = ROT180 if rot else np.eye(4)
    info = {'lods': []}
    stats = {}
    for lod in (0, 1, 2):
        nodes = []
        body = g.body(lod).transformed(R)
        fn = '%s_body_lod%d.msh' % (prefix, lod)
        body.write(os.path.join(msh_dir, fn))
        stats['body%d' % lod] = body.ntris()
        nodes += node('msh/' + fn, ['mat/body.mtl'], 'body')
        if lod == 0:
            it = g.interior()
            im = it.mesh.transformed(R)
            fn = '%s_interior_lod0.msh' % prefix
            im.write(os.path.join(msh_dir, fn))
            stats['interior'] = im.ntris()
            nodes += node('msh/' + fn, ['mat/interior.mtl'], 'interior')
            gl = g.glass().transformed(R)
            fn = '%s_glass_lod0.msh' % prefix
            gl.write(os.path.join(msh_dir, fn))
            nodes += node('msh/' + fn, ['mat/glass.mtl'], 'glass')
            info['seats'] = it.seats
            info['ilights'] = it.lights
        if lod < 2:
            for k, (dm, side, leaf, di) in enumerate(g.doors(lod)):
                dm = dm.transformed(R)
                wside = -side if rot else side
                wleaf = -leaf if rot else leaf
                fn = '%s_door%d_lod%d.msh' % (prefix, k, lod)
                dm.write(os.path.join(msh_dir, fn))
                nodes += node('msh/' + fn, ['mat/body.mtl'], 'door%d' % k, None, door_anims(wside, wleaf))
        seg = {0: 20, 1: 12, 2: 8}[lod]
        ws = parts.wheelset(g.bsw, seg, discs=not var['powered'])
        fn = 'wheelset_%s_lod%d.msh' % ('m' if var['powered'] else 't', lod)
        if not os.path.exists(os.path.join(msh_dir, fn)):
            ws.write(os.path.join(msh_dir, fn))
        info['axles'] = []
        for i, ax in enumerate(g.axles()):
            x = -ax if rot else ax
            nodes += node('msh/' + fn, ['mat/body.mtl'], 'a%d' % (i + 1), translate(x, 0, parts.AXLE_Z))
            info['axles'].append('a%d' % (i + 1))
        if g.spec.kind == 'end' and lod < 2:
            white, red = g.lights()
            fw = '%s_lights_white.msh' % prefix
            fr = '%s_lights_red.msh' % prefix
            if lod == 0:
                white.transformed(R).write(os.path.join(msh_dir, fw))
                red.transformed(R).write(os.path.join(msh_dir, fr))
            mat = ['::/vehicle/train/emissive/train_all_lights.mtl']
            if not rot:
                nodes += node('msh/' + fw, mat, 'headlights_fwd', None, light_anims('front_forward'))
                nodes += node('msh/' + fr, mat, 'taillights_bwd', None, light_anims('back_backward'))
            else:
                nodes += node('msh/' + fw, mat, 'headlights_bwd', None, light_anims('front_backward'))
                nodes += node('msh/' + fr, mat, 'taillights_fwd', None, light_anims('back_forward'))
            info['light_pos'] = [(k, (R[:3, :3] @ p) + R[:3, 3], R[:3, :3] @ n) for k, p, n in g.light_pos]
        info['lods'].append(nodes)
    info['stats'] = stats
    return info


def lights_meta(info, rot):
    out = []
    t = '\t\t\t\t\t'
    for kind, p, n in info.get('light_pos', []):
        fwd = n[0] > 0
        if kind in ('head', 'top'):
            mesh = 'headlights_bwd' if rot else 'headlights_fwd'
            col = '{ 1, 0.93, 0.78, }'
            ang, rad, st = (25, 18, 3) if kind == 'head' else (15, 12, 2)
            mode_dir = 1 if fwd else -1
            M = np.eye(4)
            M[:3, 0] = (0.98 * mode_dir, 0, -0.17)
            M[:3, 1] = (0, mode_dir, 0)
            M[:3, 2] = (0.17 * mode_dir, 0, 0.98)
            M[:3, 3] = p + n * 0.05
        else:
            mesh = 'taillights_fwd' if rot else 'taillights_bwd'
            col = '{ 0.43, 0.03, 0.03, }'
            ang, rad, st = 35, 5, 3
            mode_dir = 1 if fwd else -1
            M = np.eye(4)
            M[:3, 0] = (0.96 * mode_dir, 0, -0.26)
            M[:3, 1] = (0, mode_dir, 0)
            M[:3, 2] = (0.26 * mode_dir, 0, 0.96)
            M[:3, 3] = p + n * 0.05
        out += [t + '{ ',
                t + '\tangleMax = %d,' % ang,
                t + '\tangleMin = 5,',
                t + '\tcolor = %s,' % col,
                t + '\tfadeOutStart = 700,',
                t + '\tinvertMask = true,',
                t + '\tlightId = "",',
                t + '\tmask = 0,',
                t + '\tmeshId = "%s",' % mesh,
                t + '\tmode = "LIGHT",',
                t + '\tradius = %d,' % rad,
                t + '\tstrength = %d,' % st,
                t + '\ttransf = %s,' % lua_transf(M),
                t + '\ttype = "SPOT",',
                t + '\tvisibleFrom = -1,',
                t + '\tvisibleTo = 1000,',
                t + '},']
    for (x, z, length) in info['ilights']:
        xx = -x if rot else x
        M = np.array([[1, 0, 0, xx], [0, 0, 1, 0], [0, -1, 0, z], [0, 0, 0, 1.0]])
        out += [t + '{ ', t + '\tangleMax = 89,', t + '\tangleMin = 9,', t + '\tcolor = { 1, 0.95, 0.85, },',
                t + '\tfadeOutStart = 600,', t + '\tinvertMask = false,', t + '\tlength = %.2f,' % length,
                t + '\tlightId = "",', t + '\tmask = 32,', t + '\tmeshId = "interior",', t + '\tmode = "LIGHT",',
                t + '\tradius = 1.6,', t + '\tstrength = 3,', t + '\ttransf = %s,' % lua_transf(M),
                t + '\ttype = "CYLINDRICAL",', t + '\tvisibleFrom = -1,', t + '\tvisibleTo = 800,', t + '},']
    return out


def seats_meta(info, rot, g):
    t = '\t\t\t\t\t'
    out = []
    if g.spec.kind == 'end':
        x = 11.25
        xx = -x if rot else x
        d = -1 if rot else 1
        M = np.diag([d, d, 1, 1.0])
        M[:3, 3] = (xx, 0, rj.FLOOR_MEZZ + 0.05)
        grp = 'headlights_bwd' if rot else 'headlights_fwd'
        out += [t + '{ ', t + '\tanimation = "driving_upright",', t + '\tcrew = true,']
        if rot:
            out.append(t + '\tforward = false,')
        out += [t + '\tgroup = "%s",' % grp, t + '\ttransf = %s,' % lua_transf(M), t + '},']
    for (x, y, z, f) in info['seats']:
        if rot:
            x, y, f = -x, -y, -f
        M = np.diag([f, f, 1, 1.0])
        M[:3, 3] = (x, y, z + 0.05)
        out += [t + '{ ', t + '\tanimation = "sitting",', t + '\tgroup = "body",',
                t + '\ttransf = %s,' % lua_transf(M), t + '},']
    return out


def entrances_meta(g, rot):
    t = '\t\t\t\t\t'
    out = []
    for x in g.spec.door_xs:
        xx = -x if rot else x
        for s in (1, -1):
            pts = [(xx, 3 * s, 0.1), (xx, 2 * s, 0.1), (xx, 1.45 * s, rj.FLOOR_LOW), (xx, 0.3 * s, rj.FLOOR_LOW)]
            out += [t + '{ ', t + '\tpath = {'] + [t + '\t\t{ %.2f, %.2f, %.2f, },' % p for p in pts] + [t + '\t},', t + '},']
    return out


def labels_meta(g, rot):
    t = '\t\t\t\t\t'
    out = []
    items = []
    if g.spec.kind == 'end':
        M = np.eye(4)
        M[:3, 0] = (0, 1, 0)
        M[:3, 1] = (0, 0, 1)
        M[:3, 2] = (1, 0, 0)
        hit = g.lofts[0].surface_point(0.0, 3.95)
        x = hit[0][0] - 0.12 if hit else 12.0
        M[:3, 3] = (x, 0, 3.95)
        items.append((M, (0.95, 0.14)))
    for x in g.spec.door_xs:
        for s in (1, -1):
            M = np.eye(4)
            M[:3, 0] = (-s, 0, 0)
            M[:3, 1] = (0, 0, 1)
            M[:3, 2] = (0, s, 0)
            M[:3, 3] = (x + 1.35, s * (rj.HALF_W + 0.012), 2.78)
            items.append((M, (0.70, 0.12)))
    for M, (w, h) in items:
        if rot:
            M = ROT180 @ M
        out += [t + '{ ', t + '\talignment = "CENTER",', t + '\tchildId = "RootNode",', t + '\tcolor = { 1, 0.85, 0.2, },',
                t + '\tsize = { %.2f, %.2f, },' % (w, h), t + '\ttransf = %s,' % lua_transf(M),
                t + '\ttype = "LINE_NAME",', t + '},']
    return out


def write_mdl(g, info, rot):
    var = g.var
    spec = g.spec
    hl = spec.half_len
    lod_ranges = [(0, 220), (220, 700), (700, 3000)]
    out = ['function data()', 'return { ',
           '\t\tboundingInfo = { ',
           '\t\t\tbbMax = { %.3f, 1.5, 5.4, },' % hl,
           '\t\t\tbbMin = { %.3f, -1.5, 0, },' % -hl,
           '\t\t},',
           '\t\tcollider = { ',
           '\t\t\tparams = { ',
           '\t\t\t\thalfExtents = { %.3f, 1.41, 2.30, },' % hl,
           '\t\t\t},',
           '\t\t\ttransf = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 2.3, 1, },',
           '\t\t\ttype = "BOX",',
           '\t\t},',
           '\t\tlods = {']
    for lod, nodes in enumerate(info['lods']):
        out += ['\t\t\t{ ', '\t\t\t\tnode = { ', '\t\t\t\t\tchildren = {'] + nodes + [
            '\t\t\t\t\t},', '\t\t\t\t\tname = "RootNode",',
            '\t\t\t\t\ttransf = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, },',
            '\t\t\t\t},', '\t\t\t\tstatic = false,',
            '\t\t\t\tvisibleFrom = %d,' % lod_ranges[lod][0],
            '\t\t\t\tvisibleTo = %d,' % lod_ranges[lod][1], '\t\t\t},']
    out.append('\t\t},')
    engines = ''
    if var['powered']:
        engines = '\n'.join(['\t\t\t\t\t{ ', '\t\t\t\t\t\tpower = %d,' % POWER_KW, '\t\t\t\t\t\ttractiveEffort = %d,' % TRACTIVE_KN,
                             '\t\t\t\t\t\ttype = "ELECTRIC",', '\t\t\t\t\t},'])
    out += ['\t\tmetadata = { ',
            '\t\t\tavailability = { ', '\t\t\t\tyearFrom = 2026,', '\t\t\t\tyearTo = 0,', '\t\t\t},',
            '\t\t\tcost = { ', '\t\t\t\tprice = -1,', '\t\t\t},',
            '\t\t\tdescription = { ', '\t\t\t\tdescription = _("%s_desc"),' % var['name'],
            '\t\t\t\tname = _("%s_name"),' % var['name'], '\t\t\t},',
            '\t\t\temissions = { ',
            '\t\t\t\tnoise = { ', '\t\t\t\t\tscore = -1,', '\t\t\t\t\tidle = -1,', '\t\t\t\t\tpower = -1,', '\t\t\t\t\tspeed = -1,', '\t\t\t\t},',
            '\t\t\t\tpollution = { ', '\t\t\t\t\tscore = -1,', '\t\t\t\t\tidle = -1,', '\t\t\t\t\tpower = -1,', '\t\t\t\t\tspeed = -1,', '\t\t\t\t},',
            '\t\t\t},',
            '\t\t\textent = { ',
            '\t\t\t\tbbMax = { %.3f, 1.41, 4.6, },' % hl,
            '\t\t\t\tbbMin = { %.3f, -1.41, 0, },' % -hl,
            '\t\t\t},',
            '\t\t\tlabelList = { ', '\t\t\t\tlabels = {'] + labels_meta(g, rot) + ['\t\t\t\t},', '\t\t\t},',
            '\t\t\tlandVehicle = { ',
            '\t\t\t\tbrakeDeceleration = 1.1,',
            '\t\t\t\tcurveSpeedScale = 1,',
            '\t\t\t\tengines = {', engines, '\t\t\t\t},' if engines else '\t\t\t\t},',
            '\t\t\t\tfriction = 0.02,',
            '\t\t\t\ttopSpeed = %.4f,' % TOP_SPEED,
            '\t\t\t\tweightEmpty = %d,' % EMPTY_KG[var['kind']],
            '\t\t\t\tweightMaxPayload = %d,' % (var['capacity'] * 80),
            '\t\t\t},',
            '\t\t\tlightSourceList = { ', '\t\t\t\tlights = {'] + lights_meta(info, rot) + ['\t\t\t\t},', '\t\t\t},',
            '\t\t\tmaintenance = { ', '\t\t\t\tlifespan = 21915,', '\t\t\t\trunningCosts = -1,', '\t\t\t},',
            '\t\t\trailVehicle = { ', '\t\t\t\tconfig = { ',
            '\t\t\t\t\taxles = { %s},' % ''.join('"%s", ' % a for a in info['axles']),
            '\t\t\t\t\tfakeBogies = { },', '\t\t\t\t},', '\t\t\t},',
            '\t\t\tseatProvider = { ',
            '\t\t\t\tcrewModels = { "::/characters/era_c_driver_rail/era_c_driver_rail.mdl", },',
            '\t\t\t\tdrivingLicense = "RAIL",',
            '\t\t\t\tseats = {'] + seats_meta(info, rot, g) + ['\t\t\t\t},', '\t\t\t},',
            '\t\t\tsoundConfig = { ', '\t\t\t\tsoundSet = { ',
            '\t\t\t\t\tname = "%s",' % ('::/vehicle/train/shared/sound/train_diesel.snd'), '\t\t\t\t},', '\t\t\t},',
            '\t\t\ttransformatorConfig = { ', '\t\t\t\tskipFromLod = 2,', '\t\t\t\ttransformator = { ',
            '\t\t\t\t\tname = "::/vehicle/train/shared/default_train.trf",', '\t\t\t\t},', '\t\t\t},',
            '\t\t\ttransportVehicle = { ',
            '\t\t\t\tarrivalDelay = 2500,',
            '\t\t\t\tcarrier = "RAIL",',
            '\t\t\t\tcompartments = {', '\t\t\t\t\t{ ', '\t\t\t\t\t\tloadConfigs = {', '\t\t\t\t\t\t\t{ ',
            '\t\t\t\t\t\t\t\tcargoEntry = { ',
            '\t\t\t\t\t\t\t\t\tcapacity = %d,' % var['capacity'],
            '\t\t\t\t\t\t\t\t\tcargoTypeSet = { ',
            '\t\t\t\t\t\t\t\t\t\tcargoClassesExcluded = { },', '\t\t\t\t\t\t\t\t\t\tcargoClassesIncluded = { },',
            '\t\t\t\t\t\t\t\t\t\tcargoTypesExcluded = { },',
            '\t\t\t\t\t\t\t\t\t\tcargoTypesIncluded = { "::/cargos/passengers/passengers.cargo", },',
            '\t\t\t\t\t\t\t\t\t},', '\t\t\t\t\t\t\t\t\tloadIndicator = "",', '\t\t\t\t\t\t\t\t\tseats = { },',
            '\t\t\t\t\t\t\t\t},', '\t\t\t\t\t\t\t\ttoHide = { },', '\t\t\t\t\t\t\t},', '\t\t\t\t\t\t},', '\t\t\t\t\t},', '\t\t\t\t},',
            '\t\t\t\tdepartureDelay = 2500,',
            '\t\t\t\tengineTransportModes = { %s},' % ('"ELECTRIC_TRAIN", ' if var['powered'] else ''),
            '\t\t\t\tentrances = {'] + entrances_meta(g, rot) + ['\t\t\t\t},',
            '\t\t\t\tfilterTags = { "default", },',
            '\t\t\t\tgroupFileName = "menu_rj_dosto.mdl",',
            '\t\t\t\tloadSpeed = 3,',
            '\t\t\t\treversible = true,',
            '\t\t\t\ttransportModes = { "TRAIN", "ELECTRIC_TRAIN", },',
            '\t\t\t},',
            '\t\t},',
            '\t\tversion = 2,', '\t}', 'end', '']
    out = [l for l in out if l != '']
    with open(os.path.join(VEH, var['name'] + '.mdl'), 'w', newline='\n') as f:
        f.write('\n'.join(out) + '\n')


def write_menu_mdl():
    txt = '''function data()
return {
	boundingInfo = {
		bbMax = { 1.5, 1.5, 1.5, },
		bbMin = { -1.5, -1.5, -1.5, },
	},
	collider = {
		params = {
			halfExtents = { 1.5, 1.5, 1.5, },
		},
		transf = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, },
		type = "BOX",
	},
	lods = {
		{
			node = {
				children = {
					{
						materials = { "::/placeholders/mat/missing_object.mtl", },
						mesh = "::/placeholders/msh/cube_1x1_lod0.msh",
						name = "cube_1x1",
						transf = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, },
					},
				},
				name = "RootNode",
				transf = { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, },
			},
			static = false,
			visibleFrom = 0,
			visibleTo = 1000,
		},
	},
	metadata = {
		availability = {
			yearFrom = 2026,
			yearTo = 0,
		},
		cost = {
			price = -1,
		},
		description = {
			name = _("menu_rj_dosto"),
		},
		emissions = {
			noise = {
				score = 25,
			},
			pollution = {
				score = 0,
			},
		},
		landVehicle = {
			brakeDeceleration = 1.1,
			engines = {
				{
					power = %d,
					tractiveEffort = %d,
					type = "ELECTRIC",
				},
			},
			friction = 0.02,
			topSpeed = %.4f,
			weight = %.1f,
		},
		maintenance = {
			lifespan = 21915,
			runningCosts = -1,
		},
		railVehicle = {
			config = {
			},
		},
		transformator = {
			name = "::/vehicle/tram/shared/default_tram.trf",
		},
		transportVehicle = {
			carrier = "RAIL",
			engineTransportModes = { },
			transportModes = { },
			filterTags = { },
		},
	},
	version = 2,
	}
end
''' % (2 * POWER_KW, 2 * TRACTIVE_KN, TOP_SPEED, (2 * EMPTY_KG['end'] + 4 * EMPTY_KG['mid']) / 1000)
    with open(os.path.join(VEH, 'menu_rj_dosto.mdl'), 'w', newline='\n') as f:
        f.write(txt)


# ============================================================================ icons & preview
def scene_arrays(meshes):
    """meshes: list of (Mesh, M, kind) -> list of draw calls."""
    calls = []
    for mesh, M, kind in meshes:
        mm = mesh.transformed(M) if M is not None else mesh
        P, U, N = mm.arrays()
        calls.append((P, U, N, kind))
    return calls


def draw_calls(r, calls, tex):
    for P, U, N, kind in calls:
        if kind == 'body':
            r.draw(P, U, N, tex['alb'], tex['mga'])
        elif kind == 'interior':
            r.draw(P, U, N, tex['ialb'], None)
        elif kind == 'glass':
            r.draw(P, U, N, (0.12, 0.18, 0.22), None, alpha=0.55)
        elif kind == 'light':
            r.draw(P, U, N, (1.4, 1.35, 1.2), None)
        elif kind == 'redlight':
            r.draw(P, U, N, (0.5, 0.05, 0.05), None)


def car_draw_meshes(g, rot, lod=0, offset=0.0, with_interior=True):
    R = (ROT180 if rot else np.eye(4))
    T = translate(offset, 0, 0) @ R
    items = [(g.body(lod), T, 'body')]
    if lod == 0:
        items.append((g.glass(), T, 'glass'))
        if with_interior:
            items.append((g.interior().mesh, T, 'interior'))
    for dm, side, leaf, di in g.doors(min(lod, 1)):
        items.append((dm, T, 'body'))
    ws = parts.wheelset(g.bsw, 16, not g.var['powered'])
    for ax in g.axles():
        items.append((ws, T @ translate(ax, 0, parts.AXLE_Z), 'body'))
    if g.spec.kind == 'end':
        w, rr = g.lights()
        items.append((w, T, 'light'))
    return items


def to_tga(path, img, alpha=None, gray=False):
    if gray:
        Image.fromarray(img).convert('L').save(path)
        return
    a = Image.fromarray(img).convert('RGBA')
    if alpha is not None:
        a.putalpha(Image.fromarray(alpha))
    a.save(path)


def make_icons(geoms, tex):
    icon_dir = ensure(os.path.join(VEH, 'icons'))
    for g in geoms:
        rot = not g.var.get('cab_front', True)
        calls = scene_arrays(car_draw_meshes(g, rot, lod=1))
        name = g.var['name']
        for suffix, (w, h), eye, ortho in (('_icon20@2x', (94, 40), (0, -60, 2.3), 2.6),
                                            ('_icon_small@2x', (262, 112), (0, -60, 2.3), 2.6),
                                            ('_store', (372, 248), (12.5, -12.5, 5.5), None)):
            if ortho:
                r = Renderer(w, h, eye, (0, 0, 2.3), ortho=14.2 * h / w, ss=3)
            else:
                r = Renderer(w, h, eye, (0, 0, 1.9), fov=48, ss=3)
            draw_calls(r, calls, tex)
            img, a = r.image()
            to_tga(os.path.join(icon_dir, name + suffix + '.tga'), img, a)
            if suffix != '_store':
                to_tga(os.path.join(icon_dir, name + suffix.replace('@2x', '_cblend@2x') + '.tga'),
                       np.zeros_like(a), gray=True)
    # group icon: whole train, side view
    return


def train_layout(geoms):
    order = ['rj_dosto_1_front', 'rj_dosto_2_mid', 'rj_dosto_3_mz', 'rj_dosto_4_mid', 'rj_dosto_2_mid', 'rj_dosto_6_rear']
    byname = {g.var['name']: g for g in geoms}
    total = sum(byname[n].spec.L for n in order)
    x = total / 2
    out = []
    for n in order:
        g = byname[n]
        x -= g.spec.L / 2
        out.append((g, not g.var.get('cab_front', True), x))
        x -= g.spec.L / 2
    return out


def make_previews(geoms, tex):
    meta = ensure(os.path.join(ROOT, '_metadata'))
    layout = train_layout(geoms)
    # menu icon: side view of the full train (two rows like a catalogue)
    icon_dir = ensure(os.path.join(VEH, 'icons'))
    calls = []
    for g, rot, x in layout:
        calls += scene_arrays(car_draw_meshes(g, rot, lod=2, offset=x, with_interior=False))
    r = Renderer(520, 127, (0, -400, 2.3), (0, 0, 2.3), ortho=80.5 * 127 / 520, ss=3)
    draw_calls(r, calls, tex)
    img, a = r.image()
    to_tga(os.path.join(icon_dir, 'menu_rj_dosto_icon_small@2x.tga'), img, a)
    # 1920x1080 hero image: 3/4 front view of the leading cars
    calls = []
    for g, rot, x in layout[:3]:
        calls += scene_arrays(car_draw_meshes(g, rot, lod=0, offset=x, with_interior=True))
    front = layout[0][2] + layout[0][0].spec.half_len
    r = Renderer(1920, 1080, (front + 13.5, -13.0, 2.4), (front - 16, 0, 2.3), fov=40, ss=2, bg=(0.80, 0.86, 0.93))
    _ground(r)
    draw_calls(r, calls, tex)
    img, _ = r.image()
    _caption(img)
    Image.fromarray(img).save(os.path.join(meta, '0.png'))
    return img


def _ground(r):
    uv = lambda f, p: [(0, 0)] * 4
    m = Mesh(1)
    for i in range(-20, 20):
        for j in range(-6, 6):
            box(m, (i * 10, j * 10, -0.25), (i * 10 + 10, j * 10 + 10, -0.2), uv, faces='Z')
    P, U, N = m.arrays()
    r.draw(P, U, N, (0.55, 0.56, 0.50))
    m = Mesh(1)
    for k in range(-300, 300):
        x = k * 0.6
        box(m, (x - 0.13, -1.3, -0.24), (x + 0.13, 1.3, -0.17), uv, faces='Zxy')
    P, U, N = m.arrays()
    r.draw(P, U, N, (0.38, 0.36, 0.33))
    m = Mesh(1)
    for y in (-0.7175, 0.7175):
        for i in range(-200, 200, 2):
            box(m, (i, y - 0.035, -0.17), (i + 2, y + 0.035, 0.0), uv, faces='ZyY')
    P, U, N = m.arrays()
    r.draw(P, U, N, (0.55, 0.52, 0.50))


def _caption(img):
    from PIL import ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(paint.FONT, 26)
    d.text((30, img.shape[0] - 50), 'Render of the mod model - AI-generated (Claude)', font=f, fill=(40, 40, 40))
    img[:] = np.asarray(im)


# ============================================================================ texts
STRINGS = {
    'en': {
        'menu_rj_dosto': 'OeBB Railjet double-deck (KISS)',
        'rj_dosto_1_front_name': 'Railjet double-deck - driving trailer 1st class (front)',
        'rj_dosto_2_mid_name': 'Railjet double-deck - intermediate car 2nd class with pantograph',
        'rj_dosto_3_mz_name': 'Railjet double-deck - multi-purpose car (bikes, wheelchairs)',
        'rj_dosto_4_mid_name': 'Railjet double-deck - intermediate car 2nd class',
        'rj_dosto_6_rear_name': 'Railjet double-deck - driving trailer 2nd class (rear, turned)',
        'desc': 'Six-car double-deck long-distance multiple unit built by Stadler (KISS family) for OeBB, '
                'planned on the Vienna - Salzburg line from the end of 2026. 200 km/h, about 160 m, 486 seats. '
                'Build the train as: front driving trailer + 2nd class + multi-purpose + 2nd class + 2nd class (pantograph) + rear driving trailer. '
                'AI-generated vehicle (Claude, Anthropic); power and masses are estimates.',
    },
    'de': {
        'menu_rj_dosto': 'ÖBB Railjet Doppelstock (KISS)',
        'rj_dosto_1_front_name': 'Railjet Doppelstock - Steuerwagen 1. Klasse (vorne)',
        'rj_dosto_2_mid_name': 'Railjet Doppelstock - Mittelwagen 2. Klasse mit Stromabnehmer',
        'rj_dosto_3_mz_name': 'Railjet Doppelstock - Mehrzweckwagen (Fahrräder, Rollstühle)',
        'rj_dosto_4_mid_name': 'Railjet Doppelstock - Mittelwagen 2. Klasse',
        'rj_dosto_6_rear_name': 'Railjet Doppelstock - Steuerwagen 2. Klasse (hinten, gedreht)',
        'desc': 'Sechsteiliger Doppelstock-Fernverkehrstriebzug von Stadler (KISS-Familie) für die ÖBB, '
                'geplant ab Ende 2026 auf der Weststrecke Wien - Salzburg. 200 km/h, rund 160 m, 486 Sitzplätze. '
                'Zugbildung: Steuerwagen vorne + 2. Klasse + Mehrzweckwagen + 2. Klasse + 2. Klasse (Stromabnehmer) + Steuerwagen hinten. '
                'KI-generiertes Fahrzeug (Claude, Anthropic); Leistung und Massen sind Schätzwerte.',
    },
    'nl': {
        'menu_rj_dosto': 'ÖBB Railjet dubbeldekker (KISS)',
        'rj_dosto_1_front_name': 'Railjet dubbeldekker - stuurstandrijtuig 1e klas (voor)',
        'rj_dosto_2_mid_name': 'Railjet dubbeldekker - tussenrijtuig 2e klas met pantograaf',
        'rj_dosto_3_mz_name': 'Railjet dubbeldekker - multifunctioneel rijtuig (fietsen, rolstoelen)',
        'rj_dosto_4_mid_name': 'Railjet dubbeldekker - tussenrijtuig 2e klas',
        'rj_dosto_6_rear_name': 'Railjet dubbeldekker - stuurstandrijtuig 2e klas (achter, gedraaid)',
        'desc': 'Zesdelig dubbeldeks treinstel voor lange afstand van Stadler (KISS-familie) voor de ÖBB, '
                'gepland vanaf eind 2026 op de Westbahn Wenen - Salzburg. 200 km/h, circa 160 m, 486 zitplaatsen. '
                'Samenstelling: stuurstand voor + 2e klas + multifunctioneel + 2e klas + 2e klas (pantograaf) + stuurstand achter. '
                'AI-gegenereerd voertuig (Claude, Anthropic); vermogen en massa zijn schattingen.',
    },
}


def write_texts():
    s = {}
    for lang, d in STRINGS.items():
        s[lang] = {k: v for k, v in d.items() if k != 'desc'}
        for v in VARIANTS:
            s[lang][v['name'] + '_desc'] = d['desc']
    with open(os.path.join(ROOT, 'strings.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(s, f, ensure_ascii=False, indent=4)
    mod = {
        'dependencies': None, 'incompatibilities': None, 'modId': MOD_ID, 'params': None,
        'postRunScript': {'fileName': ''}, 'preRunScript': {'fileName': ''}, 'revision': 1,
        'runScript': {'fileName': ''}, 'severityAdd': 'None', 'severityRemove': 'Warning',
    }
    with open(os.path.join(ROOT, 'mod.json'), 'w', newline='\n') as f:
        json.dump(mod, f, indent=4)
    desc_en = open(os.path.join(os.path.dirname(__file__), 'description_en.txt'), encoding='utf-8').read()
    desc_de = open(os.path.join(os.path.dirname(__file__), 'description_de.txt'), encoding='utf-8').read()
    desc_nl = open(os.path.join(os.path.dirname(__file__), 'description_nl.txt'), encoding='utf-8').read()
    info = {
        'name': 'OeBB Railjet double-deck (Stadler KISS)',
        'summary': 'Six-car double-deck Railjet for long-distance traffic, 200 km/h. AI-generated mod.',
        'description': desc_en,
        'localization': {
            'de': {'name': 'ÖBB Railjet Doppelstock (Stadler KISS)',
                   'summary': 'Sechsteiliger Doppelstock-Railjet für den Fernverkehr, 200 km/h. KI-generierter Mod.',
                   'description': desc_de},
            'nl': {'name': 'ÖBB Railjet dubbeldekker (Stadler KISS)',
                   'summary': 'Zesdelige dubbeldeks Railjet voor lange afstand, 200 km/h. AI-gegenereerde mod.',
                   'description': desc_nl},
        },
        'authors': [{'name': 'Claude (Anthropic), for Mart Wierckx', 'role': 'CREATOR'}],
        'tags': ['Multiple Unit'],
        'url': '',
    }
    with open(os.path.join(ROOT, '_metadata', 'modinfo.json'), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
    shutil.copy(os.path.join(os.path.dirname(__file__), 'README_mod.md'), os.path.join(ROOT, 'README.md'))


# ============================================================================ main
def main():
    if os.path.exists(ROOT):
        shutil.rmtree(ROOT)
    msh_dir = ensure(os.path.join(VEH, 'msh'))
    ensure(os.path.join(ROOT, '_metadata'))
    bsw = paint.body_swatches(ATLAS)
    isw = paint.interior_swatches(INTERIOR_PX)
    geoms = [CarGeometry(v, bsw, isw) for v in VARIANTS]
    # make sure every swatch is registered before textures are baked
    for g in geoms:
        g.body(0)
        g.interior()
    print('textures ...', flush=True)
    alb, mga, ialb, glass = build_textures(geoms)
    tex = {'alb': alb, 'mga': mga, 'ialb': ialb}
    write_materials()
    write_door_anis()
    for g in geoms:
        rot = not g.var.get('cab_front', True)
        print('car', g.var['name'], flush=True)
        info = export_car(g, rot, msh_dir, g.var['name'])
        print('   ', info['stats'], 'seats', len(info['seats']))
        write_mdl(g, info, rot)
    write_menu_mdl()
    write_texts()
    print('icons ...', flush=True)
    make_icons(geoms, tex)
    make_previews(geoms, tex)
    print('done ->', ROOT)


if __name__ == '__main__':
    main()

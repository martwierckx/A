"""Uniform-colour (or small patterned) texture tiles packed into a corner of an atlas.

Small parts (bogies, seats, roof equipment, ...) map their UVs into one tile; the tile
gets painted with the part's colour and material values when the atlas is baked.
"""
import numpy as np


class Swatches:
    def __init__(self, atlas_px, region, tile_px):
        """region: (u0, v0, u1, v1) in UV units; tile_px: tile edge in pixels."""
        self.atlas_px = atlas_px
        self.region = region
        self.tile = tile_px / atlas_px
        self.cols = int(round((region[2] - region[0]) / self.tile))
        self.rows = int(round((region[3] - region[1]) / self.tile))
        self.defs = {}
        self.order = []

    def define(self, name, rgb, metal=0.0, gloss=0.4, ao=1.0, pattern=None):
        if name not in self.defs:
            idx = len(self.order)
            if idx >= self.cols * self.rows:
                raise RuntimeError('swatch region full')
            self.order.append(name)
            self.defs[name] = dict(idx=idx, rgb=rgb, metal=metal, gloss=gloss, ao=ao, pattern=pattern)
        return name

    def rect(self, name):
        i = self.defs[name]['idx']
        c, r = i % self.cols, i // self.cols
        u0 = self.region[0] + c * self.tile
        v0 = self.region[1] + r * self.tile
        return u0, v0, u0 + self.tile, v0 + self.tile

    def uvf(self, name):
        """UV function for meshlib primitives: maps every face onto the inner part of the tile."""
        u0, v0, u1, v1 = self.rect(name)
        m = 0.2 * (u1 - u0)
        a, b, c, d = u0 + m, v0 + m, u1 - m, v1 - m

        def f(face, pts):
            if face == 'cap':
                return [((a + c) / 2, (b + d) / 2), (a, d), (c, d)]
            if face == 'side':
                t0, t1 = pts
                return [(a + (c - a) * t0, d), (a + (c - a) * t1, d), (a + (c - a) * t1, b), (a + (c - a) * t0, b)]
            if face == 'ring':
                return [(a, d), (a, b), (c, b), (c, d)]
            return [(a, d), (c, d), (c, b), (a, b)]
        return f

    def paint(self, img_rgb, img_mga):
        """Fill tiles in float images (H, W, 3) with their colours / material values."""
        H = img_rgb.shape[0]
        for name in self.order:
            d = self.defs[name]
            u0, v0, u1, v1 = self.rect(name)
            x0, y0 = int(round(u0 * H)), int(round(v0 * H))
            x1, y1 = int(round(u1 * H)), int(round(v1 * H))
            rgb = np.array(d['rgb'], float)
            tile = np.broadcast_to(rgb, (y1 - y0, x1 - x0, 3)).copy()
            if d['pattern'] is not None:
                yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
                tile = d['pattern'](tile, xx / (x1 - x0), yy / (y1 - y0))
            img_rgb[y0:y1, x0:x1] = tile
            img_mga[y0:y1, x0:x1] = (d['metal'] * 255, d['gloss'] * 255, d['ao'] * 255)

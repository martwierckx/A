"""Minimal DDS (BC1/DXT1, BC3/DXT5, BC5/ATI2) encoder and BC1 decoder with mipmaps, numpy only."""
import struct
import numpy as np


def _rgb565(c):
    c = np.clip(np.rint(c), 0, 255).astype(np.uint32)
    return ((c[..., 0] >> 3) << 11) | ((c[..., 1] >> 2) << 5) | (c[..., 2] >> 3)


def _unpack565(v):
    r = ((v >> 11) & 31) * 255 / 31
    g = ((v >> 5) & 63) * 255 / 63
    b = (v & 31) * 255 / 31
    return np.stack([r, g, b], -1).astype(np.float32)


def _blocks(img, ch):
    h, w = img.shape[:2]
    return img.reshape(h // 4, 4, w // 4, 4, ch).transpose(0, 2, 1, 3, 4).reshape(-1, 16, ch).astype(np.float32)


def bc1_blocks(rgb):
    """rgb: (H, W, 3) uint8/float, H and W multiples of 4. Returns bytes."""
    blk = _blocks(rgb, 3)
    mean = blk.mean(1, keepdims=True)
    cen = blk - mean
    # principal axis via power iteration on covariance
    cov = np.einsum('nki,nkj->nij', cen, cen)
    axis = np.ones((blk.shape[0], 3), np.float32)
    for _ in range(4):
        axis = np.einsum('nij,nj->ni', cov, axis)
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    proj = np.einsum('nki,ni->nk', cen, axis)
    lo = mean[:, 0] + axis * proj.min(1)[:, None]
    hi = mean[:, 0] + axis * proj.max(1)[:, None]
    c0 = _rgb565(hi)
    c1 = _rgb565(lo)
    # ensure c0 > c1 for 4-colour mode
    swap = c0 < c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    eq = c0 == c1
    p0 = _unpack565(c0)
    p1 = _unpack565(c1)
    pal = np.stack([p0, p1, (2 * p0 + p1) / 3, (p0 + 2 * p1) / 3], 1)  # n,4,3
    d = ((blk[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)  # n,16,4
    idx = d.argmin(-1).astype(np.uint32)
    idx[eq] = 0
    bits = np.zeros(blk.shape[0], np.uint32)
    for i in range(16):
        bits |= idx[:, i] << (2 * i)
    out = np.zeros((blk.shape[0], 2), np.uint32)
    out[:, 0] = c0 | (c1 << 16)
    out[:, 1] = bits
    return out.astype('<u4').tobytes()


def bc4_blocks(ch):
    """ch: (H, W) single channel. Returns 8 bytes per block."""
    blk = _blocks(ch[..., None], 1)[..., 0]
    a0 = np.rint(blk.max(1)).astype(np.int64)
    a1 = np.rint(blk.min(1)).astype(np.int64)
    flat = a0 == a1
    a0 = np.where(flat & (a0 < 255), a0 + 1, a0)
    a1 = np.where(flat & (a0 == 255) & (a1 == a0), a1 - 1, a1)
    pal = np.zeros((blk.shape[0], 8), np.float32)
    pal[:, 0] = a0
    pal[:, 1] = a1
    for i in range(1, 7):
        pal[:, i + 1] = ((7 - i) * a0 + i * a1) / 7
    idx = np.abs(blk[:, :, None] - pal[:, None, :]).argmin(-1).astype(np.uint64)
    bits = np.zeros(blk.shape[0], np.uint64)
    for i in range(16):
        bits |= idx[:, i] << np.uint64(3 * i)
    out = np.zeros((blk.shape[0], 8), np.uint8)
    out[:, 0] = a0
    out[:, 1] = a1
    for i in range(6):
        out[:, 2 + i] = ((bits >> np.uint64(8 * i)) & np.uint64(255)).astype(np.uint8)
    return out


def _mips(img):
    levels = [img.astype(np.float32)]
    while levels[-1].shape[0] > 4 and levels[-1].shape[1] > 4:
        a = levels[-1]
        levels.append(0.25 * (a[0::2, 0::2] + a[1::2, 0::2] + a[0::2, 1::2] + a[1::2, 1::2]))
    return levels


def _header(w, h, mips, fourcc, linsize):
    flags = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000 | 0x80000
    hdr = struct.pack('<4sIIIIIII', b'DDS ', 124, flags, h, w, linsize, 0, mips)
    hdr += b'\0' * 44
    hdr += struct.pack('<II4sIIIII', 32, 0x4, fourcc, 0, 0, 0, 0, 0)
    hdr += struct.pack('<IIIII', 0x1000 | 0x400000 | 0x8, 0, 0, 0, 0)
    return hdr


def save(path, img, fmt):
    """img: (H, W, C) array in 0..255. fmt: 'DXT1' (RGB), 'DXT5' (RGBA), 'ATI2' (RG)."""
    img = np.asarray(img, np.float32)
    h, w = img.shape[:2]
    levels = _mips(img)
    data = b''
    for lv in levels:
        if fmt == 'DXT1':
            data += bc1_blocks(lv[..., :3])
        elif fmt == 'DXT5':
            a = bc4_blocks(lv[..., 3]).reshape(-1, 8)
            c = np.frombuffer(bc1_blocks(lv[..., :3]), np.uint8).reshape(-1, 8)
            data += np.concatenate([a, c], 1).tobytes()
        elif fmt == 'ATI2':
            r = bc4_blocks(lv[..., 0])
            g = bc4_blocks(lv[..., 1])
            data += np.concatenate([r, g], 1).tobytes()
        else:
            raise ValueError(fmt)
    bpb = 8 if fmt == 'DXT1' else 16
    with open(path, 'wb') as f:
        f.write(_header(w, h, len(levels), fmt.encode(), (w // 4) * (h // 4) * bpb))
        f.write(data)


def load_bc1(path):
    """Decode the top mip of a DXT1 file to (H, W, 3) uint8."""
    b = open(path, 'rb').read()
    h, w = struct.unpack('<II', b[12:20])
    n = (w // 4) * (h // 4)
    blk = np.frombuffer(b[128:128 + n * 8], '<u4').reshape(n, 2)
    c0, c1 = blk[:, 0] & 0xFFFF, blk[:, 0] >> 16
    p0, p1 = _unpack565(c0), _unpack565(c1)
    four = (c0 > c1)[:, None]
    p2 = np.where(four, (2 * p0 + p1) / 3, (p0 + p1) / 2)
    p3 = np.where(four, (p0 + 2 * p1) / 3, 0)
    pal = np.stack([p0, p1, p2, p3], 1)
    idx = (blk[:, 1][:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3
    px = pal[np.arange(n)[:, None], idx]
    return px.reshape(h // 4, w // 4, 4, 4, 3).transpose(0, 2, 1, 3, 4).reshape(h, w, 3).astype(np.uint8)

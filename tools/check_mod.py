"""Sanity checks for a built mod: Lua syntax, mesh blobs, referenced files."""
import glob
import os
import re
import sys

import numpy as np
from lupa import LuaRuntime

root = sys.argv[1]
L = LuaRuntime()
L.execute('_ = function(s) return s end')
errors = 0
veh = glob.glob(os.path.join(root, 'content', 'vehicle', 'train', '*'))[0]
for f in glob.glob(os.path.join(veh, 'msh', '*.msh')):
    L.execute(open(f).read())
    d = L.globals().data()
    b = open(f + '.blob', 'rb').read()
    va = {k: (v['count'], v['offset'], v['numComp']) for k, v in d.vertexAttr.items()}
    for k, (c, o, n) in va.items():
        if o + c > len(b) or c % (4 * n):
            print('bad attr', f, k); errors += 1
    for sm in d.subMeshes.values():
        for k, v in sm.indices.items():
            c, o = v['count'], v['offset']
            idx = np.frombuffer(b[o:o + c], '<u4')
            nverts = va[k][0] // (4 * va[k][2])
            if len(idx) == 0 or idx.max() >= nverts or len(idx) % 3:
                print('bad index', f, k); errors += 1
for f in glob.glob(os.path.join(veh, '*.mdl')):
    src = open(f).read()
    L.execute(src)
    L.globals().data()
    for ref in re.findall(r'"((?:msh|mat|ani)/[^"]+)"', src):
        if not os.path.exists(os.path.join(veh, ref)):
            print('missing', f, ref); errors += 1
for f in glob.glob(os.path.join(veh, 'mat', '*.mtl')):
    src = open(f).read()
    L.execute(src)
    for ref in re.findall(r'"(tex/[^"]+)"', src):
        if not os.path.exists(os.path.join(veh, 'mat', ref)):
            print('missing', f, ref); errors += 1
print('errors:', errors)

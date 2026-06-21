import sys, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from skin_tone import _to_lab_d65, _get_undertone_refs
from colormath.color_diff import delta_e_cie2000

cases = [
    ('Light cool (235,200,205)', (235,200,205), 2),
    ('Medium cool (180,155,150)', (180,155,150), 5),
    ('Light cool true (220,190,210)', (220,190,210), 2),
]
for desc, rgb, monk in cases:
    lab  = _to_lab_d65(tuple(x/255.0 for x in rgb))
    refs = _get_undertone_refs(monk)
    d_w  = delta_e_cie2000(lab, refs['warm'])
    d_c  = delta_e_cie2000(lab, refs['cool'])
    d_n  = delta_e_cie2000(lab, refs['neutral'])
    wref = refs['warm']
    cref = refs['cool']
    nref = refs['neutral']
    print(f'{desc}:')
    print(f'  Pixel L*={lab.lab_l:.2f} a*={lab.lab_a:.2f} b*={lab.lab_b:.2f}')
    print(f'  Warm  ref L*={wref.lab_l:.1f} a*={wref.lab_a:.1f} b*={wref.lab_b:.1f}  dE={d_w:.2f}')
    print(f'  Cool  ref L*={cref.lab_l:.1f} a*={cref.lab_a:.1f} b*={cref.lab_b:.1f}  dE={d_c:.2f}')
    print(f'  Neut  ref L*={nref.lab_l:.1f} a*={nref.lab_a:.1f} b*={nref.lab_b:.1f}  dE={d_n:.2f}')
    best = min(d_w, d_c, d_n)
    second = sorted([d_w, d_c, d_n])[1]
    print(f'  Gap (2nd-best - best): {second-best:.2f}  margin=1.8  -> ', end='')
    if second-best < 1.8:
        print('Neutral (gap too small)')
    elif best == d_w: print('Warm')
    elif best == d_c: print('Cool')
    else: print('Neutral')
    print()

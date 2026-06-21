import sys, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')
from skin_tone import MONK_TONES_HEX, _hex_to_rgb, MONK_L_D65, classify_tone, classify_undertone

print('=== TEST 1: PURE MONK PALETTE ===')
exact = within1 = 0
for i, h in enumerate(MONK_TONES_HEX):
    r,g,b = _hex_to_rgb(h)
    px = np.tile([r,g,b],(300,1)).astype(np.uint8)
    tone, conf, mL = classify_tone(px, np.array([r,g,b],float), window=2)
    ok = 'EXACT' if tone==i+1 else ('W1' if abs(tone-i-1)<=1 else 'MISS')
    if tone==i+1: exact+=1
    if abs(tone-i-1)<=1: within1+=1
    print(f'  [{ok:5s}] Monk {i+1:2d}  pred={tone}  conf={conf:.0%}')
print(f'  Exact={exact}/10  Within-1={within1}/10')

print()
print('=== TEST 2: WARM LIGHT (+10R, -8B) ===')
errs = []
for i in range(10):
    r,g,b = _hex_to_rgb(MONK_TONES_HEX[i])
    r2=min(255,int(r*1.10)); b2=max(0,int(b*0.92))
    px = np.tile([r2,g,b2],(300,1)).astype(np.uint8)
    tone, conf, _ = classify_tone(px, np.array([r2,g,b2],float), window=2)
    e = abs(tone-(i+1)); errs.append(e)
    print(f'  Monk {i+1:2d} -> pred={tone}  err={tone-i-1:+d}  conf={conf:.0%}')
print(f'  Mean abs err: {sum(errs)/10:.2f}')

print()
print('=== TEST 3: DARK SKIN SHADOW (-10V) ===')
errs = []
for i in range(6,10):
    r,g,b = _hex_to_rgb(MONK_TONES_HEX[i])
    r2=max(0,int(r*0.90)); g2=max(0,int(g*0.90)); b2=max(0,int(b*0.90))
    px = np.tile([r2,g2,b2],(300,1)).astype(np.uint8)
    tone, conf, mL = classify_tone(px, np.array([r2,g2,b2],float), window=2)
    e = abs(tone-(i+1)); errs.append(e)
    print(f'  Monk {i+1:2d} (-10V) -> pred={tone}  err={tone-i-1:+d}  conf={conf:.0%}')
print(f'  Mean abs err: {sum(errs)/4:.2f}')

print()
print('=== TEST 4: UNDERTONE (7 cases) ===')
cases = [
    ('Light warm (pinkish-gold)',  (240,195,170), 2, 'Warm'),
    ('Light cool (rosy-pink-blue)',(220,190,210), 2, 'Cool'),
    ('Light neutral',              (235,200,188), 2, 'Neutral'),
    ('Medium warm (golden)',       (195,148, 95), 5, 'Warm'),
    ('Medium cool (ashy)',         (172,155,165), 5, 'Cool'),
    ('Dark warm (warm brown)',     ( 85, 55, 35), 8, 'Warm'),
    ('Dark cool (cool deep)',      ( 62, 55, 65), 9, 'Cool'),
    ('Dark neutral',               ( 70, 55, 50), 9, 'Neutral'),
]
correct = 0
for desc, rgb, monk, exp in cases:
    px = np.tile(list(rgb),(200,1)).astype(np.uint8)
    ut, conf = classify_undertone(px, monk)
    ok = 'OK' if ut==exp else 'XX'
    if ut==exp: correct+=1
    print(f'  [{ok}] {desc:32s}: pred={ut:7s} exp={exp:7s} conf={conf:.0%}')
print(f'  Result: {correct}/{len(cases)} correct')

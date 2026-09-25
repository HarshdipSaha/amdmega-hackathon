"""Generate synthetic dev and holdout splits for ROADREAD Task 13."""
import hashlib, json, random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEV_OUT = Path('eval/data/dev')
HOLD_OUT = Path('eval/data/holdout')
rng = random.Random(42)

CN_PROVINCES = '京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁'
CN_LETTERS = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
CN_SERIALS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
WORD_SIGNS = ['STOP','YIELD','ONE WAY','NO ENTRY','DO NOT ENTER','WRONG WAY','KEEP RIGHT','KEEP LEFT','NO PASSING','NO TURN ON RED','NO LEFT TURN','NO RIGHT TURN','NO U TURN','NO PARKING','TOW AWAY ZONE','SCHOOL ZONE','BUS STOP','BIKE LANE','PEDESTRIAN CROSSING','DEAD END']
SPEEDS = [15,20,25,30,35,40,45,50,55,65]
ADVISORY = [10,15,20,25,30,35,40,45,50,55]
WARNING_SIGNS = ['ROAD WORK AHEAD','SLOW MEN WORKING','FLAGMAN AHEAD','DETOUR AHEAD','BUMP AHEAD','DIP AHEAD','SLIPPERY WHEN WET','NARROW BRIDGE','LOW CLEARANCE','LANE ENDS MERGE LEFT','LANE ENDS MERGE RIGHT','DIVIDED HIGHWAY BEGINS','DIVIDED HIGHWAY ENDS','TWO WAY TRAFFIC','CROSS TRAFFIC DOES NOT STOP','EMERGENCY SIGNAL AHEAD','SHARP CURVE AHEAD','WINDING ROAD AHEAD','HILL AHEAD','STEEP GRADE AHEAD']

def _plate_us(r):
    chars = 'ABCDEFGHJKLMNPRSTUVWXYZ'
    d = '0123456789'
    pats = [
        lambda: f"{r.choice(d)}{r.choice(chars)}{r.choice(chars)}{r.choice(d)}{r.choice(d)}{r.choice(d)}",
        lambda: f"{r.choice(chars)}{r.choice(chars)}{r.choice(chars)}{r.choice(d)}{r.choice(d)}{r.choice(d)}",
        lambda: f"{r.choice(chars)}{r.choice(chars)}{r.choice(d)}{r.choice(d)}{r.choice(chars)}{r.choice(chars)}",
        lambda: f"{r.choice(d)}{r.choice(d)}{r.choice(d)}{r.choice(chars)}{r.choice(chars)}{r.choice(chars)}",
    ]
    return r.choice(pats)()

def _plate_cn(r):
    prov = r.choice(CN_PROVINCES)
    letter = r.choice(CN_LETTERS)
    length = r.choice([5,5,5,6])
    serial = ''.join(r.choice(CN_SERIALS) for _ in range(length))
    if length == 6:
        serial = list(serial); serial[r.choice([0,-1])] = r.choice('DF'); serial = ''.join(serial)
    return f"{prov}{letter}·{serial}"

def _get_font(size, cjk=False):
    paths = ['msyh.ttc', 'simsun.ttc', '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'] if cjk else []
    paths += ['arial.ttf','Arial.ttf','arialbd.ttf','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf','/usr/share/fonts/TTF/DejaVuSans-Bold.ttf']
    for path in paths:
        try: return ImageFont.truetype(path, size)
        except: pass
    return ImageFont.load_default()

def _draw_centered(img, text, font_size=60, color=(0,0,0), cjk=False):
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size, cjk=cjk)
    words = text.split()
    if len(words) > 3:
        mid = len(words)//2
        lines = [' '.join(words[:mid]), ' '.join(words[mid:])]
    else:
        lines = [text]
    line_h = font_size + 8
    total_h = len(lines) * line_h
    y = (img.height - total_h) // 2
    for line in lines:
        try:
            bbox = draw.textbbox((0,0), line, font=font)
            tw = bbox[2] - bbox[0]
        except:
            tw = len(line) * (font_size // 2)
        x = (img.width - tw) // 2
        draw.text((x, y), line, font=font, fill=color)
        y += line_h

def _render(slice_name, text, r):
    if slice_name == 'us_plate':
        bg = r.choice([(255,255,255),(240,240,220),(220,230,255)])
        img = Image.new('RGB',(500,220),bg)
        _draw_centered(img, text, font_size=88, color=(0,0,0))
    elif slice_name == 'cn_plate':
        img = Image.new('RGB',(520,200),(0,51,153))
        _draw_centered(img, text, font_size=68, color=(255,255,255), cjk=True)
    elif slice_name == 'word_sign':
        bg = (190,0,0) if text=='STOP' else (255,255,255)
        fg = (255,255,255) if text=='STOP' else (0,0,0)
        img = Image.new('RGB',(480,480),bg)
        _draw_centered(img, text, font_size=70, color=fg)
    elif slice_name == 'speed_sign':
        img = Image.new('RGB',(480,560),(255,255,255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([12,12,468,548],outline=(0,0,0),width=8)
        _draw_centered(img, text, font_size=68, color=(0,0,0))
    elif slice_name == 'advisory_plaque':
        img = Image.new('RGB',(300,300),(255,220,0))
        draw = ImageDraw.Draw(img)
        draw.polygon([150,8,292,150,150,292,8,150],outline=(0,0,0),width=6)
        _draw_centered(img, text, font_size=110, color=(0,0,0))
    else:  # warning_sign
        img = Image.new('RGB',(500,500),(255,200,0))
        draw = ImageDraw.Draw(img)
        draw.rectangle([12,12,488,488],outline=(0,0,0),width=8)
        _draw_centered(img, text, font_size=54, color=(0,0,0))
    return img.resize((640,480), Image.LANCZOS)

def _degrade(img, deg, r):
    if deg=='clean': return img
    if deg=='blur': return img.filter(ImageFilter.GaussianBlur(radius=r.uniform(1.5,3.0)))
    if deg=='noise':
        px=list(img.getdata())
        noisy=[tuple(max(0,min(255,c+int(r.gauss(0,20)))) for c in p) for p in px]
        out=Image.new('RGB',img.size); out.putdata(noisy); return out
    if deg=='low_light': return img.point(lambda x: int(x*r.uniform(0.3,0.55)))
    if deg=='glare':
        out=img.copy(); draw=ImageDraw.Draw(out)
        cx,cy=r.randint(60,580),r.randint(60,420); rad=r.randint(25,60)
        draw.ellipse([cx-rad,cy-rad,cx+rad,cy+rad],fill=(255,255,255))
        return out
    if deg=='skew':
        w,h=img.size; sk=r.uniform(0.04,0.12)
        coeffs=(1,sk,-sk*h/2,0,1,0,0,0)
        return img.transform(img.size,Image.PERSPECTIVE,coeffs,Image.BILINEAR)
    if deg=='motion_blur':
        return img.filter(ImageFilter.GaussianBlur(radius=1.5))
    return img

DEGS=['clean','blur','noise','low_light','glare','skew','motion_blur']
SLICES=['us_plate','cn_plate','word_sign','speed_sign','advisory_plaque','warning_sign']

def _text_gen(slice_name, r):
    if slice_name=='us_plate': return _plate_us(r)
    if slice_name=='cn_plate': return _plate_cn(r)
    if slice_name=='word_sign': return r.choice(WORD_SIGNS)
    if slice_name=='speed_sign': return f"SPEED LIMIT {r.choice(SPEEDS)}"
    if slice_name=='advisory_plaque': return str(r.choice(ADVISORY))
    return r.choice(WARNING_SIGNS)

def _sha256(path):
    h=hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

def generate(out_dir, prefix, n=20):
    out_dir.mkdir(parents=True,exist_ok=True)
    rows=[]; idx=0
    for sl in SLICES:
        for i in range(n):
            text = _text_gen(sl, rng)
            deg = DEGS[i % len(DEGS)]
            img = _render(sl, text, rng)
            img = _degrade(img, deg, rng)
            ext = ['png','jpg','tiff'][idx % 3]
            fname = f"{sl}_{idx:03d}.{ext}"
            fpath = out_dir / fname
            if ext=='jpg': img.save(fpath,'JPEG',quality=82)
            elif ext=='tiff': img.save(fpath,'TIFF',compression='tiff_lzw')
            else: img.save(fpath,'PNG')
            rows.append({'image':fname,'gold':text,'slice':sl,'degradation':deg,'family':f"{prefix}{sl}_{i}","sha256":_sha256(fpath)})
            idx+=1
    mf = out_dir / f"{out_dir.name}.jsonl"
    mf.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    total_mb=sum(f.stat().st_size for f in out_dir.iterdir() if f.suffix in ('.png','.jpg','.tiff','.jsonl'))/1e6
    print(f"{out_dir}: {len(rows)} images, {total_mb:.1f} MB")
    return rows

if __name__=='__main__':
    d=generate(DEV_OUT,'dev_')
    h=generate(HOLD_OUT,'hold_')
    # Verify no family overlap
    dev_fam={r['family'] for r in d}
    hold_fam={r['family'] for r in h}
    assert not dev_fam & hold_fam, f"Family overlap: {dev_fam & hold_fam}"
    print(f"No family overlap. Dev={len(d)}, Holdout={len(h)}")

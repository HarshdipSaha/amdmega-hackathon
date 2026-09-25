from PIL import Image
from roadread.decode import load_rgb, bounded, center_crop, reread_view

def _save(tmp_path, name, img, **kw):
    p = tmp_path / name; img.save(p, **kw); return p

def test_formats_and_modes(tmp_path):
    for name, img in [("a.png", Image.new("P", (40, 20))), ("b.jpg", Image.new("L", (40, 20))),
                      ("c.tiff", Image.new("RGBA", (40, 20))), ("d.tif", Image.new("I;16", (40, 20)))]:
        im = load_rgb(_save(tmp_path, name, img))
        assert im.mode == "RGB" and im.size == (40, 20)

def test_transparent_becomes_white(tmp_path):
    im = load_rgb(_save(tmp_path, "t.png", Image.new("RGBA", (4, 4), (0, 0, 0, 0))))
    assert im.getpixel((0, 0)) == (255, 255, 255)

def test_exif_orientation(tmp_path):
    img = Image.new("RGB", (40, 20)); ex = img.getexif(); ex[0x0112] = 6
    assert load_rgb(_save(tmp_path, "r.jpg", img, exif=ex)).size == (20, 40)

def test_multipage_tiff_first_frame(tmp_path):
    a, b = Image.new("RGB", (10, 10), "red"), Image.new("RGB", (30, 30), "blue")
    p = tmp_path / "m.tiff"; a.save(p, save_all=True, append_images=[b])
    assert load_rgb(p).size == (10, 10)

def test_bounded_preserves_aspect_and_never_upscales():
    out = bounded(Image.new("RGB", (4000, 1000)), max_pixels=1_000_000)
    assert out.width * out.height <= 1_000_000 and abs(out.width / out.height - 4.0) < 0.02
    assert bounded(Image.new("RGB", (100, 50)), 1_000_000).size == (100, 50)

def test_reread_view_differs_from_first_view():
    assert reread_view(Image.new("RGB", (300, 100)), 1_000_000, 2_500_000).size == (600, 200)   # small: 2x up
    v = reread_view(Image.new("RGB", (4000, 1000)), 1_000_000, 2_500_000)                       # big: more pixels
    assert 1_000_000 < v.width * v.height <= 2_500_000

def test_center_crop_fraction():
    assert center_crop(Image.new("RGB", (100, 100)), 0.8).size == (80, 80)

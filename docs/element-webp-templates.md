# Element WEBP Template Notes

The OCR backend currently loads element templates from `backend/Data/Elements/*.png`.
This is a loader convention, not an OpenCV limitation: `cv2.imdecode` can read the
Encore `IconElementAttri` WEBP assets correctly.

## Current Runtime Behavior

- `data.load_templates("Elements", ...)` only scans `*.png`.
- Dropping a `.webp` file into `backend/Data/Elements` will be ignored until the
  loader is extended.
- Existing element recognition uses the echo panel crop from `card.py`, then
  `get_element_region()` to isolate the badge.
- `determine_element()` uses HSV histogram matching first, then SIFT only when
  top candidates share a hue cluster.

## WEBP Test Result

Test image:

```text
C:\Users\domin\Downloads\347056b9395d3315667093a2532b153a13d9160d.jpeg
```

Local WEBP tested:

```text
C:\Users\domin\Downloads\T_IconElementAttriIceA1.webp
```

Result: `T_IconElementAttriIceA1.webp` decoded as a `76x76` image and matched
the Hiyuki / Wishes of Quiet Snowfall badge correctly across all five echo
panels. The backend's normal PNG templates also resolved all five panels as
`QuietSnow`.

Representative scores from the same badge crop path:

```text
echo1 server=QuietSnow | IceA1 HSV=0.9792 color=0.4226 SIFT=0.1014
echo2 server=QuietSnow | IceA1 HSV=0.9733 color=0.2286 SIFT=0.0145
echo3 server=QuietSnow | IceA1 HSV=0.9609 color=0.5291 SIFT=0.0290
echo4 server=QuietSnow | IceA1 HSV=0.9709 color=0.5361 SIFT=0.0725
echo5 server=QuietSnow | IceA1 HSV=0.9799 color=0.4323 SIFT=0.0435
```

## Encore Asset Notes

The Encore path tested was:

```text
https://api-v2.encore.moe/resource/Data/Game/Aki/UI/UIResources/Common/Image/IconElementAttri/T_IconElementAttriAdam.webp
```

The directory itself does not expose a browsable index. Direct asset requests
work, but some clients need a browser-like `User-Agent`; Python `urllib` without
one received `403`, while PowerShell `Invoke-WebRequest` returned `200`.

Likely filenames that resolved during probing:

```text
T_IconElementAttriAdam.webp
T_IconElementAttriAttack.webp
T_IconElementAttriDark.webp
T_IconElementAttriFire.webp
T_IconElementAttriIce.webp
T_IconElementAttriIceA1.webp
T_IconElementAttriLight.webp
T_IconElementAttriWind.webp
```

For the QuietSnow badge, `IceA1` was the useful match. Plain `Ice` had a strong
coarse HSV score because the hue is similar, but it did not match the badge shape
well by color-template score or SIFT.

## Recommended Next Test

To test WEBP templates in runtime, either:

- Convert selected WEBPs to PNG and place them in `backend/Data/Elements` with
  canonical backend names like `QuietSnow.png`.
- Or extend `load_templates()` to scan both `*.png` and `*.webp`, then add an
  explicit filename-to-canonical-element mapping for Encore asset names such as
  `T_IconElementAttriIceA1 -> QuietSnow`.

The second option is better for a direct CDN sync, but it should be done with a
checked mapping table. The current backend logic expects template keys to be
canonical element names (`QuietSnow`, `Glacio`, `Frosty`, etc.), not raw asset
filenames.

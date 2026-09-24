# ABCurves showcase source

This directory builds the self-contained ABCurves website from its HTML, CSS, JavaScript and frozen motion data. The build uses Python 3.10 or later and only its standard library. It needs no model weights, research checkout, package installation or network connection.

```sh
python web/build.py
node web/qa/verify.cjs docs/index.html
python -m http.server 8765 --bind 127.0.0.1 --directory docs
```

Open `http://127.0.0.1:8765` for a local preview. The generated `docs/index.html` contains all styling, application code and motion data. The adjacent `.nojekyll` supports static serving. Nothing is published by these commands.

Use `python web/build.py --output path/to/index.html` to choose another output location. The Node check accepts the generated HTML path as an optional argument. Node is needed only for those additional checks, not for building or viewing the page.

## Source files

| File | Purpose |
| --- | --- |
| `src/index.html` | Accessible page structure and four build placeholders |
| `src/styles.css` | Desktop and mobile layouts |
| `src/app.js` | Playback, comparisons, camera, navigation and controls |
| `data/catalog.json` | Readable titles, metadata, thumbnails and ordered example IDs |
| `data/motion.json.gz` | Frozen JSON containing the 50 exact compressed motion blocks |
| `data/selection.json` | Continuous order and renderer comparison selection |
| `data/provenance.json` | Original site hash, source identities, model and payload hashes, renderer profiles/seeds, and GIF receipts |
| `../assets/continuous_roaming.gif` | H18 README animation |
| `../assets/continuous_switchbacks.gif` | H17 README animation |
| `qa/verify.cjs` | Portable data, framing, gesture and playback checks |

The motion JSON is ordinary UTF-8 JSON stored in gzip to keep the repository compact. `python web/build.py --dump-motion motion-inspection.json` writes a readable copy. Each record retains its original base64 zlib block and typed-array layout; rebuilding does not recompress or resample the motion. The arrays use little-endian integer or floating-point values at eight-byte-aligned offsets. Generated and recorded hardware reports remain integers at their original 1 ms cadence.

Edit the HTML, CSS, JavaScript or catalog text and run `python web/build.py` to rebuild. `--verify-approved` additionally checks byte-for-byte equality with the reference page identified in provenance. Ordinary builds verify each motion payload and GIF independently of the page text.

## What the examples mean

There are 24 original Static Planner examples, 18 selected Continuous Planner examples and eight Renderer comparisons. Continuous/H17 opens first. The continuous page retains its fixed order, four draws, 1× playback, 1.2-second trail, target guide and following camera. Static and Renderer default to 0.5×. Playback repeats automatically. Phone controls remain available through View; desktop wheel and pinch zoom preserve following, while desktop panning switches to manual control.

The renderer comparisons show genuine human reports alongside renderings of that same human's smoothed movement. Each of these comparisons has 256 genuine preceding reports. The Static fixtures use a 160-report prefix plus a 96-report quiet-start assumption. These distinct source conditions are recorded in provenance.

The continuous visualizations are curated demonstrations, including training-source recordings, with their original raw-history initialization and rendered outputs. Planner and Renderer sampling seeds are recorded separately. Target-path guides are visual aids; generation receives causal target observations.

The page is rebuilt from frozen demonstration outputs. Source hashes, splits, groups and component identities connect the examples to the training and generation workflows.

## Verification

The Python build validates all 50 payloads, checks compressed and decoded checksums, array intervals and finite values, and verifies both GIFs. The optional Node check decodes every payload, exercises the actual camera functions at every millisecond across five panel layouts, and checks twelve navigation behaviors plus elapsed-time playback and automatic repeat. It writes `qa/verification.json`.

The page needs Canvas, Pointer Events, native dialogs and `DecompressionStream`; preview it with a current browser through the local server.

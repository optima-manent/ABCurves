# Public runtime checks

From the repository after installing `.[all,dev]`:

```bash
python -m pytest tests/public_runtime -q
```

The tests exercise actual Continuous inference and persistent native rendering,
including history preparation, coordinate conversion, target receipt timing, reset,
state isolation and partial-output failure handling. Synthetic cases isolate API
boundaries.

`fixtures/human_start.json` binds the genuine session 6099 fixture to its raw archive,
episode, sample location, coordinate transform and attribution. The fixture contains
256 native reports, 164 common displacements and their 160 filtered history samples.

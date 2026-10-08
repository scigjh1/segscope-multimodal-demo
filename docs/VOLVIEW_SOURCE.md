# Viewer source

Source: https://github.com/Kitware/VolView
Official demo: https://volview.kitware.app/
Pinned release: v4.5.2
Commit: 90d8a81a354c3d64732a5f2372bdedcf020b7032
License: Apache-2.0 (web/volview/LICENSE)

The complete upstream viewer, volume-rendering controls, image browser, measurements and segmentation tools are preserved. Changes are limited to the SegScope identifier, page title and local GPU/report entry points. The original VolView logo, About box and author/license credits remain. Error reporting has no configured DSN.

The launcher serves a generated geometric NRRD phantom and proxies an already configured local GPU runtime. The phantom contains no patient data. GPU model architecture, adapter and checkpoints are stored outside this repository.

## Build and launch

Use Node 22.12+ (verified here with Node 24.19).

```bash
cd web/volview
npm ci
npm run build
cd ../..
python scripts/serve_workbench.py --port 4200 --runtime http://127.0.0.1:4186
```

Open http://127.0.0.1:4200. Model inference needs the separately configured GPU runtime; the viewer itself loads local files without that runtime.

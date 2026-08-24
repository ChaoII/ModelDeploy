# Test Model Migration (yolo26n + ppocrv6_tiny) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate all unit tests and regression baselines from the yolo11n family + ppocrv4 to the yolo26n family + ppocrv6_tiny, regenerate baselines, and verify locally (ort/mnn/trt) and on the sophgo device.

**Architecture:** Path-swap test model references, drop the `_nms`/`_without_nms`/pre-raw regression variants (yolo26n ships end2end NMS), regenerate committed baseline JSONs with `tests/baseline_collect.exe`, then run the Catch2 `test_modeldeploy` binary on each backend. Sophgo goes through the `.243` cross-compile container → `.70` device pipeline in `docs/sophgo_cross_build_and_test.md`.

**Tech Stack:** C++17, Catch2 single binary, pybind11/onnxruntime/MNN/TensorRT, sophgo bmrt (device side).

## Global Constraints

- Branch `capi-v2`; Chinese commit messages.
- New canonical model files (all exist on disk under `test_data/test_models/`):
  - onnx: `onnx/yolo26n/{yolo26n,yolo26n-cls,yolo26n-seg,yolo26n-pose,yolo26n-obb}.onnx`
  - mnn: `mnn/yolo26n/{yolo26n,yolo26n-cls,yolo26n-seg,yolo26n-pose,yolo26n-obb}.mnn`
  - trt: `trt/yolo26n/{yolo26n,yolo26n-cls,yolo26n-seg,yolo26n-pose,yolo26n-obb}.engine`
  - sophgo: `sophgo/yolo26n/{yolo26n,yolo26n-seg,yolo26n-pose,yolo26n-obb}-int8|-f16.bmodel` (rename from `_INT8`/`_F16`)
  - OCR: `onnx|mnn/ocr/ppocrv6_tiny/{det,cls,rec}_infer.*`, dict `ppocrv6_tiny_dict.txt`
- **Sophgo naming rule (user-mandated):** short dash + lowercase quant type: `<prefix>-int8.bmodel` / `<prefix>-f16.bmodel`.
- Drop `_nms` / `_without_nms` / pre/raw regression variants; keep end-to-end det/seg/pose/obb/cls iters.
- Baselines live at `tests/baselines/{ort,mnn,trt,sophgo}/`, committed, `<model-filename>.<ext>.<type>.json`.
- `TEST_DATA_DIR` is set by CMake to `${CMAKE_SOURCE_DIR}`; `get_test_data()` → `<repo>/test_data`; `baseline_root()` → `<repo>/tests/baselines`.
- Test binary: `build/bin/test_modeldeploy.exe`. Baseline collector: `build/bin/baseline_collect.exe`.
- Do NOT touch SDK inference logic; keep behavioral assertion style (`size>0`, `label>=0`).
- Build (already configured, ORT+MNN+TRT+GPU): `cmake --build build --target test_modeldeploy baseline_collect`.

---

### Task 1: Migrate `tests/test_vision_models.cpp` paths to yolo26n + ppocrv6_tiny

**Files:**
- Modify: `tests/test_vision_models.cpp`

**Interfaces:**
- Consumes: existing model classes unchanged (`Classification`, `UltralyticsDet/Seg/Pose/Obb`, `DBDetector`, `Classifier`, `Recognizer`).
- Produces: same test names/semantics, now exercising yolo26n + ppocrv6_tiny.

- [ ] **Step 1: Replace every yolo11n path with its yolo26n equivalent (`onnx/yolo11n/yolo11n` → `onnx/yolo26n/yolo26n`).** The strings to replace (all occurrences):
  - `onnx/yolo11n/yolo11n-cls.onnx` → `onnx/yolo26n/yolo26n-cls.onnx`
  - `onnx/yolo11n/yolo11n.onnx` → `onnx/yolo26n/yolo26n.onnx`
  - `onnx/yolo11n/yolo11n-seg.onnx` → `onnx/yolo26n/yolo26n-seg.onnx`
  - `onnx/yolo11n/yolo11n-pose.onnx` → `onnx/yolo26n/yolo26n-pose.onnx`
  - `onnx/yolo11n/yolo11n-obb.onnx` → `onnx/yolo26n/yolo26n-obb.onnx`

- [ ] **Step 2: Swap OCR default to ppocrv6_tiny** — in the three OCR tests change preference order so `ppocrv6_tiny` is checked first and v4/v5 become fallbacks:
  - `onnx/ocr/ppocrv4_mobile/det_infer.onnx` → try `onnx/ocr/ppocrv6_tiny/det_infer.onnx` first, then `ppocrv5_mobile/det_infer.onnx` as fallback.
  - Same for `cls_infer.onnx` and `rec_infer.onnx`.
  - `get_test_data() / "ppocrv4_dict.txt"` → `get_test_data() / "ppocrv6_tiny_dict.txt"`.

- [ ] **Step 3: Build.**

```
cmake --build build --target test_modeldeploy baseline_collect
```
Expected: builds with no errors.

- [ ] **Step 4: Run the touched suites.**

From `build/bin`:
```
./test_modeldeploy.exe "[vision_models]" 2>&1 | tail -20
```
Expected: `All tests passed` (any `(.)`/skips are fine — a test returns early if its model file is absent).

- [ ] **Step 5: Commit.**

```bash
git add tests/test_vision_models.cpp
git commit -m "test(vision_models): migrate to yolo26n + ppocrv6_tiny"
```

---

### Task 2: Migrate `tests/test_capi.cpp` paths to yolo26n + ppocrv6_tiny

**Files:**
- Modify: `tests/test_capi.cpp`

**Interfaces:**
- Consumes: C API unchanged (`md_model_create`, `md_model_predict`, etc.).
- Produces: CAPI model param/`predict` tests now run on yolo26n + ppocrv6_tiny.

- [ ] **Step 1: Replace these exact strings across the file:**
  - `test_models/onnx/yolo11n/yolo11n.onnx` → `test_models/onnx/yolo26n/yolo26n.onnx`
  - `test_models/onnx/yolo11n/yolo11n-cls.onnx` → `test_models/onnx/yolo26n/yolo26n-cls.onnx`
  - `test_models/onnx/yolo11n/yolo11n-pose.onnx` → `test_models/onnx/yolo26n/yolo26n-pose.onnx`
  - `test_models/onnx/ocr/ppocrv4_mobile` → `test_models/onnx/ocr/ppocrv6_tiny`
  - `test_data/ppocrv4_dict.txt` → `test_data/ppocrv6_tiny_dict.txt`

- [ ] **Step 2: Build + run `[capi]`, `[model]` suites.**

```
cmake --build build --target test_modeldeploy
cd build/bin
./test_modeldeploy.exe "[capi]" 2>&1 | tail -10
./test_modeldeploy.exe "[model]" 2>&1 | tail -10
```
Expected: `All tests passed` (or skips where models absent).

- [ ] **Step 3: Commit.**

```bash
git add tests/test_capi.cpp
git commit -m "test(capi): migrate to yolo26n + ppocrv6_tiny"
```

---

### Task 3: `tests/test_pipelines.cpp` prefer ppocrv6_tiny dict

**Files:**
- Modify: `tests/test_pipelines.cpp` (function `ocr_dict()` at ~line 34)

**Interfaces:**
- Consumes: `ocr::PaddleOCR` unchanged; `find_ocr_model()` already probes `ppocrv6_tiny`.
- Produces: OCR pipeline test resolves to v6_tiny model + dict.

- [ ] **Step 1: Update `ocr_dict()` search list** so v6_tiny dict is found first. Current:
```cpp
for (const auto& d : {"ppocrv4_dict.txt", "ppocrv5_dict.txt", "dict.txt"}) {
```
Change to:
```cpp
for (const auto& d : {"ppocrv6_tiny_dict.txt", "ppocrv4_dict.txt", "ppocrv5_dict.txt", "dict.txt"}) {
```

- [ ] **Step 2: Build + run `[pipeline]`.**

```
cmake --build build --target test_modeldeploy
cd build/bin
./test_modeldeploy.exe "[pipeline]" 2>&1 | tail -10
```
Expected: `All tests passed` (OCR, LPR, face, insightface). Skips allowed where a model is missing.

- [ ] **Step 3: Commit.**

```bash
git add tests/test_pipelines.cpp
git commit -m "test(pipelines): prefer ppocrv6_tiny dict"
```

---

### Task 4: Rewrite `tests/baseline_compare.cpp` to yolo26n (drop `_nms` / pre/raw variants)

**Files:**
- Modify: `tests/baseline_compare.cpp`

**Interfaces:**
- Consumes: `baseline_utils` serialize/compare funcs unchanged; `baseline_collect` output naming `<filename>.<type>.json`.
- Produces: end-to-end regression TEST_CASEs for det/seg/pose/obb/cls across ort/mnn/trt/sophgo, plus OCR(det/rec/cls) and scrfd on ort.

**Step 1 through 4 are the mechanical rewrite.** Keep these existing Test-Case skeletons (paths changed), and **delete** every case whose name contains `_nms`, `with/without_nms`, or that calls `compare_yolo_pre_raw` / `check_tensors` with `pre.json`/`raw.json`.

- [ ] **Step 1: Replace model path strings** (same mapping as Task 1, plus backend dirs):
  - `yolo11n.onnx` → `yolo26n.onnx`, `yolo11n-cls.onnx` → `yolo26n-cls.onnx`, `yolo11n-seg.onnx` → `yolo26n-seg.onnx`, `yolo11n-pose.onnx` → `yolo26n-pose.onnx`, `yolo11n-obb.onnx` → `yolo26n-obb.onnx`
  - `.mnn` and `.engine` equivalents likewise.
  - `ocr/ppocrv4_mobile/det_infer.onnx` → `ocr/ppocrv6_tiny/det_infer.onnx`; same for `cls_infer`/`rec_infer`; `ppocrv4_dict.txt` → `ppocrv6_tiny_dict.txt`.

- [ ] **Step 2: Delete the variant/pre-raw cases.** Remove these TEST_CASEs entirely:
  - `Regression: yolo11n detection + pre/raw` (the one calling `compare_yolo_pre_raw`)
  - `Regression: yolo11n_*_nms ...` (detection/obb/pose/seg) for onnx/mnn/trt/sophgo
  - `Regression: yolo11n-seg ...` doubling with seg_nms only where it's the duplicate — keep one end-to-end seg, one end-to-end pose, one end-to-end obb per backend (drop `_nms` twins).
  - `yolo11n detection Sophgo` etc. → become `yolo26n` with `-int8`/`-f16` bmodel paths (see Task 8 for naming).

- [ ] **Step 3: Verify no stale refs.** Grep should show zero `_nms`, `without_nms`, `ppocrv4`, `pre.json`, `raw.json` in the file:
```
Select-String -Path tests\baseline_compare.cpp -Pattern '_nms|without_nms|ppocrv4|pre\.json|raw\.json'
```
Expected: no output.

- [ ] **Step 4: Build (compile-only check).**
```
cmake --build build --target test_modeldeploy
```
Expected: builds with no errors (test not run yet — baselines regenerated in Task 5).

- [ ] **Step 5: Commit.**
```bash
git add tests/baseline_compare.cpp
git commit -m "test(regression): baseline_compare -> yolo26n, drop _nms/pre-raw variants"
```

---

### Task 5: Delete old baselines, regenerate ORT baselines

**Files:**
- Modify: `tests/baselines/ort/*.json`, `tests/baselines/mnn/*.json`, `tests/baselines/trt/*.json`, `tests/baselines/sophgo/*.json` (delete old, add new)

**Interfaces:**
- Consumes: `baseline_collect.exe --backend ort|mnn|trt`.
- Produces: committed baseline JSONs that `baseline_compare` reads by `<filename>.<type>.json`.

- [ ] **Step 1: Delete all existing baseline JSONs but keep `.gitkeep` + dirs.**
```
Get-ChildItem tests\baselines -Recurse -Filter *.json | Remove-Item
```

- [ ] **Step 2: Regenerate ORT baselines** (names derive from model filename, so they match what `baseline_compare` loads). From `build/bin`, run one per model/type. Test images: `test_detection0.jpg`, `test_person.jpg`, `test_ocr.png`.
```
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\yolo26n\yolo26n.onnx --image ..\..\test_data\test_images\test_detection0.jpg --out ..\..\tests\baselines\ort --type det --backend ort --family det
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\yolo26n\yolo26n-seg.onnx --image ..\..\test_data\test_images\test_person.jpg --out ..\..\tests\baselines\ort --type seg --backend ort --family seg
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\yolo26n\yolo26n-pose.onnx --image ..\..\test_data\test_images\test_person.jpg --out ..\..\tests\baselines\ort --type pose --backend ort --family pose
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\yolo26n\yolo26n-obb.onnx --image ..\..\test_data\test_images\test_detection0.jpg --out ..\..\tests\baselines\ort --type obb --backend ort --family obb
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\yolo26n\yolo26n-cls.onnx --image ..\..\test_data\test_images\test_person.jpg --out ..\..\tests\baselines\ort --type cls --backend ort --family cls
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\ocr\ppocrv6_tiny\det_infer.onnx --image ..\..\test_data\test_images\test_ocr.png --out ..\..\tests\baselines\ort --type ocr_det --backend ort
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\ocr\ppocrv6_tiny\rec_infer.onnx --image ..\..\test_data\test_images\test_ocr.png --out ..\..\tests\baselines\ort --type ocr_rec --backend ort
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\ocr\ppocrv6_tiny\cls_infer.onnx --image ..\..\test_data\test_images\test_ocr.png --out ..\..\tests\baselines\ort --type ocr_cls --backend ort
.\baseline_collect.exe --model ..\..\test_data\test_models\onnx\face\scrfd_2.5g_bnkps_shape640x640.onnx --image ..\..\test_data\test_images\test_face.jpg --out ..\..\tests\baselines\ort --type face_det --backend ort
```
Expected: 10 JSON files appear under `tests/baselines/ort/` named e.g. `yolo26n.onnx.det.json`, `yolo26n-cls.onnx.cls.json`, `det_infer.onnx.ocr_det.json`, etc.

- [ ] **Step 3: Regenerate MNN baselines** (same files, `--backend mnn`, out `tests/baselines/mnn`), using the `.mnn` models:
```
./baseline_collect.exe --model ..\..\test_data\test_models\mnn\yolo26n\yolo26n.mnn ... --out ..\..\tests\baselines\mnn --type det --backend mnn --family det
. # repeat for seg/pose/obb/cls with mnn models
```
- [ ] **Step 4: Regenerate TRT baselines** (`--backend trt`, `.engine` models, out `tests/baselines/trt`). NOTE: TRT runs on GPU; if `[gpu]` build is present this works; otherwise defer to a GPU-capable run.
- [ ] **Step 5: Commit ORT/MNN/TRT baselines.**
```bash
git add tests/baselines
git commit -m "test: regenerate baselines for yolo26n + ppocrv6_tiny (ort/mnn/trt)"
```

---

### Task 6: Run local regression + full local suites (ort/mnn/trt)

**Files:** (none — verification)

- [ ] **Step 1: Run `[regression]` and confirm self-vs-baseline `require_no_diff` + cross-backend `warn_diff` (warn ok, fail not).**
```
cd build/bin
./test_modeldeploy.exe "[regression]" 2>&1 | tail -25
```
Expected: `All tests passed`. A `[WARN]` cross-backend diff is acceptable; a `FAILED` is not.

- [ ] **Step 2: Full local run excluding GPU-dependent cases that can't run here this session; then targeted GPU runs.**
```
./test_modeldeploy.exe "~[sophgo]" 2>&1 | tail -15
./test_modeldeploy.exe "[gpu]" 2>&1 | tail -15   # ort_gpu, trt, mnn_cuda
```
Expected: `All tests passed` (or documented skips where a TRT/MNN/GPU artifact is absent).

---

### Task 7: Sophgo — rename bmodels `-int8`/`-f16`, update refs, cross-compile, deploy, regenerate sophgo baselines, run on device

**Files:**
- Modify: `test_data` bmodel files and every sophgo path reference (tests + `examples/demo_*/*_sophgo*.cpp` + `gen_demos.py`)

**Interfaces:**
- Consumes: SOP in `docs/sophgo_cross_build_and_test.md` (.243 container `tpuc_dev` / `/workspace/build_sophgo`, device `.70` / `linaro:linaro`).
- Produces: `tests/baselines/sophgo/*.json` for yolo26n; sophgo regression passes on device.

- [ ] **Step 1: On the local test_data tree, rename sophgo yolo26n bmodels** `_F16`→`-f16`, `_INT8`→`-int8` (e.g. `yolo26n-seg_INT8.bmodel` → `yolo26n-seg-int8.bmodel`). Update every reference (`examples/demo_*/*sophgo*.cpp`, `gen_demos.py`, and Task 4's sophgo baseline_compare cases) to the new names.
- [ ] **Step 2: Cross-compile in `.243` container** (per SOP §3.2, `ENABLE_SOPHGO=ON ENABLE_ORT=OFF`) the `test_modeldeploy` + `baseline_collect` targets (or their sophgo equivalents) and the updated demos.
- [ ] **Step 3: Deploy** `test_modeldeploy`, `baseline_collect`, and the renamed bmodels + `ppocrv6_tiny` mnn? (not needed) to `.70` `build_sophgo/bin` (SOP §3.3).
- [ ] **Step 4: On device, regenerate sophgo baselines** with `baseline_collect --backend sophgo` for yolo26n det/seg/pose/obb/cls (`-int8`/`-f16`), writing `tests/baselines/sophgo/`. Pull JSONs back to the repo.
- [ ] **Step 5: On device, run `[regression]` `[backend:sophgo]`** and confirm pass. Fetch results back.
- [ ] **Step 6: Commit baselines + sophgo ref/naming changes.**
```bash
git add test_data/examples tests examples gen_demos.py 2>/dev/null; git add tests/baselines/sophgo
git commit -m "test(sophgo): yolo26n baselines + -int8/-f16 naming"
```

---

### Task 8: Final verification & cleanup

**Files:** — verification

- [ ] **Step 1: Full `ctest` gate on GPU build** (CMAKE_BUILD_TYPE Release).
```
cmake --build build && cd build && ctest -C Release --output-on-failure
```
Expected: all pass (exclude `[sophgo]` tags locally).
- [ ] **Step 2: Remove generated runtime artifacts** (`result_*.jpg`, `*.log`) under `build/bin`.
- [ ] **Step 3: Confirm `git status` clean aside from intended commits; `git log --oneline -10` shows the migration commits.**

# Model Clone — All Binding Layers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add `clone()` to every C++ model class, then expose through CAPI, pybind, C#, Rust.

**Architecture:** Simple models (single model file, preproc/postproc members) use `make_unique<Model>(*this) + set_runtime(clone_runtime())`. Pipeline models (hold sub-models via unique_ptr/shared_ptr) deep-clone each sub-model. CAPI uses a single generic `md_clone_model()` that dispatches on `MDModelType`. Higher layers call CAPI.

**Tech Stack:** C++17, C API, pybind11, C#, Rust (manual FFI)

## Global Constraints

- Every C++ model gets `clone()` declared in its header and defined in its .cpp
- Pipeline models must deep-clone all sub-model unique_ptr/shared_ptr members
- Kokoro (TTS) is excluded — too complex (jieba, voice binary, text normalizer)
- CAPI clone function returns `MDStatusCode` and fills an output `MDModel`
- C# `Clone()` returns the model wrapper type
- Rust `try_clone()` returns `Result<Self, MdError>`
- Python `clone()` returns a new model instance

---

### Task 1: C++ clone() — Simple Models (Part 1)

**Files to modify:**
- `csrc/vision/classification/classification.h` + `.cpp`
- `csrc/vision/iseg/ultralytics_seg.h` + `.cpp`
- `csrc/vision/obb/ultralytics_obb.h` + `.cpp`
- `csrc/vision/pose/ultralytics_pose.h` + `.cpp`

**Pattern (each model):**
```cpp
// Header: add declaration
[[nodiscard]] std::unique_ptr<ModelType> clone() const;

// .cpp: add implementation
std::unique_ptr<ModelType> ModelType::clone() const {
    auto clone_model = std::make_unique<ModelType>(*this);
    clone_model->set_runtime(clone_model->clone_runtime());
    return clone_model;
}
```

---

### Task 2: C++ clone() — Simple Models (Part 2)

**Files to modify:**
- `csrc/vision/face/face_rec/seetaface.h` + `.cpp` (SeetaFaceID)
- `csrc/vision/face/face_gender/seetaface_gender.h` + `.cpp` (SeetaFaceGender)
- `csrc/vision/face/face_age/seetaface_age.h` + `.cpp` (SeetaFaceAge)
- `csrc/vision/lpr/lpr_det/lpr_det.h` + `.cpp` (LprDetection)
- `csrc/vision/lpr/lpr_rec/lpr_rec.h` + `.cpp` (LprRecognizer)

Same pattern as Task 1.

---

### Task 3: C++ clone() — Simple Models (Part 3, face_as)

- `csrc/vision/face/face_as/face_as_first.h` + `.cpp` (SeetaFaceAsFirst — no preproc/postproc members, just clone runtime + copy size_)
- `csrc/vision/face/face_as/face_as_second.h` + `.cpp` (SeetaFaceAsSecond — same)

---

### Task 4: C++ clone() — Pipeline / Composite Models

- **PaddleOCR**: `csrc/vision/ocr/ppocr.h` + `.cpp` — deep-clone detector_, classifier_, recognizer_
- **FaceRecognizerPipeline**: `csrc/vision/face/face_rec_pipeline/face_rec_pipeline.h` + `.cpp` — deep-clone detector_, recognizer_
- **SeetaFaceAsPipeline**: `csrc/vision/face/face_as/face_as_pipeline.h` + `.cpp` — deep-clone face_det_, face_as_first_, face_as_second_
- **LprPipeline**: `csrc/vision/lpr/lpr_pipeline/lpr_pipeline.h` + `.cpp` — deep-clone detector_, recognizer_
- **PedestrianAttribute**: `csrc/vision/pipeline/pedestrian_attribute.h` + `.cpp` — deep-clone detector_, classifier_
- **PPStructureV2Table**: `csrc/vision/ocr/ppstructurev2_table.h` + `.cpp` — deep-clone detector_, recognizer_, table_

**Pipeline clone pattern:**
```cpp
std::unique_ptr<PipelineModel> PipelineModel::clone() const {
    auto clone_model = std::make_unique<PipelineModel>(*this);
    // Deep-clone each sub-model
    if (sub_model_a_) clone_model->sub_model_a_ = sub_model_a_->clone();
    if (sub_model_b_) clone_model->sub_model_b_ = sub_model_b_->clone();
    // Replace runtime for this pipeline's own runtime
    clone_model->set_runtime(clone_model->clone_runtime());
    return clone_model;
}
```

Note: `make_unique<PipelineModel>(*this)` with shared_ptr sub-models would share them. The deep-clone lines above replace them.

Note: For `PedestrianAttribute` which uses `shared_ptr`, the copy constructor already creates aliasing shared_ptrs. We must `.reset(new_sub_model)` to make them independent.

---

### Task 5: C++ clone() — DBDetector, Classifier, Recognizer (sub-models used internally)

These are BaseModel subclasses used internally by PaddleOCR and PPStructureV2Table. They need clone() for their parent pipeline models to deep-clone them.

- `csrc/vision/ocr/dbdetector.h` + `.cpp` — clone() (simple model)
- `csrc/vision/ocr/classifier.h` + `.cpp` — clone() (simple model)

(Recognizer already covered in Task 1 or 2)

---

### Task 6: CAPI — Add md_clone_model

**Files:**
- `capi/vision/md_model_capi.h` (NEW or add to existing capi header)
- `capi/vision/md_model_capi.cpp` (NEW)

**Single generic clone function:**
```c
MODELDEPLOY_CAPI_EXPORT MDStatusCode md_clone_model(MDModel* model, const MDModel* from);
```

Implementation switches on `from->type`:
```cpp
MDStatusCode md_clone_model(MDModel* model, const MDModel* from) {
    switch (from->type) {
        case Detection: {
            auto* src = static_cast<UltralyticsDet*>(from->model_content);
            model->model_content = src->clone().release();
            break;
        }
        case Classification: {
            auto* src = static_cast<Classification*>(from->model_content);
            model->model_content = src->clone().release();
            break;
        }
        // ... etc for all types
        default: return ModelTypeError;
    }
    return Success;
}
```

---

### Task 7: pybind — Add clone() to all Python bindings

**File:** `csrc/pybind/`

For each model pybind class, add:
```cpp
.def("clone", [](const ModelType& self) {
    return self.clone();
})
```

---

### Task 8: C# — Add Clone() to all wrapper classes

**File:** `csharp/ModelDeploy/`

For each model wrapper class, add:
```csharp
public ModelType Clone() {
    var clone = new MDModel();
    var status = md_clone_model(ref clone, ref _model);
    CheckStatus(status);
    return new ModelType(clone);
}
```

---

### Task 9: Rust — Add try_clone() to all structs

**File:** `rust/modeldeploy/src/vision/*.rs`

For each model struct, add:
```rust
pub fn try_clone(&self) -> Result<Self, MdError> {
    let mut clone = ffi::MDModel { ... };
    let status = unsafe { ffi::md_clone_model(&mut clone, &self.model) };
    check_status(status)?;
    Ok(Self { model: clone })
}
```

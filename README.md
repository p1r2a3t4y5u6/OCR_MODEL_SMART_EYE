# OCR Pipeline (OpenCV + Tesseract)

A small, dependency-light Python OCR pipeline built for the technical
challenge: *"Build a small Python-based OCR pipeline that takes an image
as input, preprocesses it, extracts text and returns the extracted
output."*

It also incorporates and documents a second artifact included in this
repo: [`notebooks/CIG_MODEL.ipynb`](notebooks/CIG_MODEL.ipynb), a
deep-learning (CNN + Transformer, CTC loss) model trained to read short
alphanumeric codes off noisy cigarette-pack images. That notebook's
preprocessing research is reused here as an optional mode — see
[Two preprocessing modes](#two-preprocessing-modes) below.

## Approach

**Backend:** OpenCV for preprocessing, Tesseract (via `pytesseract`) for
text extraction. Both are free, run entirely offline/on-CPU, and need no
GPU or trained weights — appropriate for a "small pipeline" deliverable.

**Pipeline stages** (`src/pipeline.py`):

```
image (path) → cv2.imread → preprocess() → pytesseract → OCRResult(text, confidence)
```

### Two preprocessing modes

| Mode | When to use | Steps |
|---|---|---|
| `document` (default) | Screenshots, receipts, scanned pages, signage — general text | grayscale → upscale if small → denoise (`fastNlMeansDenoising`) → adaptive Gaussian threshold → deskew (`minAreaRect` on foreground pixels) |
| `code` | Short, noisy printed codes (e.g. batch/date codes) | grayscale → dilate → erode (morphological closing) → Otsu threshold |

`code` mode reuses the exact morphological recipe (`dilate_image`,
`erode_image`, `compute_gradient`) copied out of
`notebooks/CIG_MODEL.ipynb`, Section 3b. That notebook documents (Section
3a) *why* this recipe was chosen: plain morphology cleans up speckle
noise but can't fully separate text from large ink-blob artifacts;
contour-based blob removal and an ESRGAN+SAM segmentation attempt both
failed for the same reason, which is why the notebook ultimately trains
a CTC model that gets a raw gradient channel as extra signal instead of
trying to fully clean the image. `src/preprocessing.py` re-implements
the notebook's cleaned-image channel (not the gradient/model part) as a
standalone step so it's usable outside of the training notebook.

Adaptive thresholding is used instead of a single global (Otsu)
threshold in `document` mode because it holds up much better under the
uneven lighting typical of a phone photo or a screenshot with soft
shadows.

## Repo layout

```
.
├── notebooks/
│   └── CIG_MODEL.ipynb        # deep-learning OCR model + preprocessing R&D (reference)
├── src/
│   ├── preprocessing.py       # OpenCV preprocessing (both modes)
│   ├── ocr_engine.py          # pytesseract wrapper
│   └── pipeline.py            # OCRPipeline class + CLI
├── samples/
│   ├── input/
│   │   ├── sample_1.png           # sample input #1 (document mode)
│   │   └── sample_2_code.png       # sample input #2 (code mode, synthetic)
│   └── output/
│       ├── sample_1_result.json
│       ├── sample_1_output.txt
│       ├── sample_1_preprocessed.png
│       ├── sample_2_result.json
│       └── sample_2_preprocessed.png
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <this-repo-url>
cd ocr-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Tesseract binary (not a Python package) must also be installed:
sudo apt-get install tesseract-ocr        # Debian/Ubuntu
brew install tesseract                    # macOS
```

## Usage

```bash
# Plain text to stdout
python -m src.pipeline samples/input/sample_1.png

# Full JSON (text, confidence, word count, timing)
python -m src.pipeline samples/input/sample_1.png --json

# Short/noisy code crop, and save the preprocessed debug image
python -m src.pipeline samples/input/sample_2_code.png --mode code --save-debug out.png
```

As a library:

```python
from src.pipeline import OCRPipeline

pipeline = OCRPipeline(mode="document")
result = pipeline.run("samples/input/sample_1.png")
print(result.text, result.mean_confidence)
```

Run tests:

```bash
python -m pytest tests/ -v
```

## Sample input/output

### Sample 1 — document mode

Input (`samples/input/sample_1.png`) is a screenshot of the challenge
brief itself:

![sample 1 input](samples/input/sample_1.png)

Preprocessed image actually fed to Tesseract
(`samples/output/sample_1_preprocessed.png`):

![sample 1 preprocessed](samples/output/sample_1_preprocessed.png)

Extracted text (`samples/output/sample_1_output.txt`, real pipeline
output, mean word confidence **91.08%**):

```
Complete the following optional technical challenge: Task: Build a small Python-based OCR Pipeline that takes an image as input, preprocesses it, extracts text and returns the extracted output. You may use Opencv + Tesseract/PaddleOCR/EasyOCR or another framework. Submit: GitHub repository Short README Sample input/output Brief explanation of your approach . share link Your answer TT ) This is a required question
```

(Two-character-level slips — "OCR pipeline" → "OCR Pipeline", "OpenCV" →
"Opencv" — are typical, expected Tesseract case/spacing noise; every
word is correctly recognized.)

### Sample 2 — code mode

Input (`samples/input/sample_2_code.png`) is a synthetic noisy code
crop — clean text plus speckle noise and an ink-blob artifact,
deliberately built to mirror the failure case described in
`CIG_MODEL.ipynb`:

![sample 2 input](samples/input/sample_2_code.png)

After morphological closing (`samples/output/sample_2_preprocessed.png`)
— note the speckle is gone but the ink blob survives, exactly the
limitation the notebook documents:

![sample 2 preprocessed](samples/output/sample_2_preprocessed.png)

Extracted text: `"Bus22%"` (ground truth: `"BU522X"`).

This is left in deliberately, unedited, as an honest result: generic
Tesseract on a tiny, noisy 6-character crop with a surviving blob
artifact is exactly the scenario `CIG_MODEL.ipynb` was built to solve
with a custom-trained model instead of a generic OCR engine — this repo
is not claiming the classical `code` mode matches that trained model,
only that it reuses its preprocessing.

## Design decisions & limitations

- **Why Tesseract over PaddleOCR/EasyOCR:** zero extra model downloads,
  CPU-only, smallest dependency footprint — fits "small pipeline."
  Swapping engines only requires implementing the same
  `extract_text()`/`extract_data()` interface in `src/ocr_engine.py`.
- **Confidence score** is the mean of Tesseract's per-word confidences
  (`image_to_data`), not a single global heuristic.
- **Known limitation:** `code` mode's classical thresholding cannot fully
  separate text from large ink-blob artifacts (see Sample 2 above) — this
  is exactly why `CIG_MODEL.ipynb` trains a learned model with a gradient
  channel instead of relying on preprocessing alone for that dataset.
- **Not included:** the trained model/weights from `CIG_MODEL.ipynb` —
  that notebook depends on a private dataset (`cig_ps.zip` on Google
  Drive) that isn't part of this repo; it's included here purely as
  supporting research documentation for the `code` preprocessing mode.

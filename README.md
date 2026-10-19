# segmenter-tiny – Diplomatic Segmentation of Medieval Charters

Splits a **medieval Latin or Middle High German charter** into its diplomatic parts
(*invocatio, intitulatio, arenga, publicatio, narratio, dispositio, sanctio, corroboratio, subscriptio, datatio, apprecatio*)
with a compact token-classification model.

| | |
|---|---|
| Model | [`ERCDiDip/charter-diplomatic-segmentation`](https://huggingface.co/ERCDiDip/charter-diplomatic-segmentation) (CC BY-NC 4.0, DOI [10.57967/hf/10291](https://doi.org/10.57967/hf/10291)) |
| Architecture | `BertForTokenClassification` on [Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384), 117 M parameters, 22 BIO labels (11 sections × B/I, no outside class) |
| Training | Knowledge distillation from an XLM-RoBERTa-base teacher (278 M) that was trained on CEI-annotated charters from Monasterium.net; see *How it works* |
| Test performance | Lambach hold-out, 94 charters: token accuracy **81.6 %**, span F1 (IoU ≥ 0.5) **76.8 %** (teacher: 81.0 % / 75.5 %) |

## Contents

| File | Purpose |
|---|---|
| `charter_segmenter.py` | Ready-to-use script / Python module: downloads the model and segments text (CLI, JSON, HTML output) |
| `charter_segmenter_demo.ipynb` | Demo notebook (executed, with outputs): the label scheme, four charters, confidence plot, smoothing, windowing |
| `Otto_965_IV_03_FreisBm_diploma.txt` | Otto I grants property to Negomir, vassal of Bishop Abraham of Freising (Monasterium.net FreisBm 0965_IV_03) |
| `PannHActa_1416_VIII_10_charter.txt` | Sigismund to Abbot Demetrius of Tihany, 1416 Aug 10 (Latin, Hungarian royal chancery) |
| `KarlIV_Strassburg_mandate_mhg.txt` | Charles IV to the city of Strasbourg (Middle High German mandate, issued at Prague) |
| `Barletta_1417_XII_05_notarial_instrument.txt` | Notarial instrument from Barletta, 1417 Dec 5 (Latin, 1,717 words: tests the sliding window) |

## Installation

```bash
pip install torch transformers huggingface_hub
pip install jupyter matplotlib pandas   # only for the notebook
```

The first run downloads the model (470 MB, cached afterwards). A GPU is used automatically if available;
on a CPU the 1,717-word charter takes about two seconds.

## Usage

### Command line

```bash
python charter_segmenter.py Otto_965_IV_03_FreisBm_diploma.txt                 # sections on stdout
python charter_segmenter.py charter.txt --json out.json --html view.html       # save results
python charter_segmenter.py charter.txt --words                                # one line per word: label, confidence
python charter_segmenter.py charter.txt --min-span 0                           # raw spans, no smoothing
python charter_segmenter.py --text "In nomine sancte et individue trinitatis. Ego ..."
```

Options: `--min-span` (default 5, see *Smoothing*), `--device cpu|cuda`, `--model` (Hub id or local folder).

Output for the Otto diploma with `--min-span 0`:

```
[INVOCATIO] (7 words, conf 0.59) C. In nomine sanctae et indiuiduae trinitatis.
[INTITULATIO] (6 words, conf 0.93) Otto diuina annuente clementia imperator augustus.
[PUBLICATIO] (8 words, conf 0.94) Nouerint omnes fideles nostri praesentes scilicet et futuri,
[NARRATIO] (25 words, conf 0.81) qualiter nos per interuentum dilecte ducis (!) domineque Judite, ...
[DISPOSITIO] (86 words, conf 1.00) proprietatem, qualem nos visi sumus habere ad Vuirzsosah ...
[CORROBORATIO] (18 words, conf 0.98) Et vt haec nostre donationis auetoritas firma permaneat, ...
[SUBSCRIPTIO] (13 words, conf 0.49) Signum domni Ottonis (M) pigissimi (!) imperatoris.Liutulfus notarius ...
[DATATIO] (22 words, conf 0.97) Data iii. nonas Apriles anno vero dominice incarnationis DCCCC.LXV., ...
[APPRECATIO] (1 words, conf 0.29) amen.
```

### Python

```python
from charter_segmenter import CharterSegmenter

seg = CharterSegmenter()
spans = seg.predict("In nomine sancte et individue trinitatis. Ego Lupoldus dei gratia dux Austrie ...")
for sp in spans:
    print(sp["section"], sp["n_words"], round(sp["conf"], 2), sp["text"])

words = seg.predict_words(text)      # one dict per word: word, label (B-/I-), section, conf, char offsets
```

Every span is a dict with `section`, `text`, `start_char`, `end_char`, `start_word`, `end_word`, `n_words`
and `conf` (mean word confidence). `to_html(text, spans)` renders a colour-highlighted page.

### Notebook

```bash
jupyter notebook charter_segmenter_demo.ipynb
```

### Without the wrapper

The checkpoint is a standard `BertForTokenClassification`, so plain `transformers` code works
(labels must be read from `model.config.id2label`, whose order is alphabetical):

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
tok = AutoTokenizer.from_pretrained("ERCDiDip/charter-diplomatic-segmentation")
model = AutoModelForTokenClassification.from_pretrained("ERCDiDip/charter-diplomatic-segmentation")
enc = tok(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True)
labels = model(**enc).logits[0].argmax(-1)
```

## Sections

`INVOCATIO` invocation of God · `INTITULATIO` name and title of the issuer · `ARENGA` preamble ·
`PUBLICATIO` announcement to the public · `NARRATIO` circumstances of the act · `DISPOSITIO` the legal act itself ·
`SANCTIO` penalty clause · `CORROBORATIO` means of authentication · `SUBSCRIPTIO` subscriptions and witnesses ·
`DATATIO` date and place · `APPRECATIO` closing wish.

Every word receives exactly one section (there is no outside class). Parts that the scheme does not contain,
such as the *inscriptio* and *salutatio* of a mandate, are absorbed by the neighbouring section, usually `INTITULATIO`.

## How it works

The teacher is an XLM-RoBERTa-base token classifier trained on charters with CEI structural annotation
(Monasterium.net). The student, Multilingual-MiniLM-L12-H384 with a linear head, was trained with token-level
knowledge distillation (temperature-scaled KL to the teacher's soft labels), a cross-entropy term on the teacher's
hard labels and a class-weighted cross-entropy on the annotated spans. The student follows the didip annotation
convention, which keeps `NARRATIO` distinct from `DISPOSITIO`.

### Decoding

`charter_segmenter.py` reproduces the decoding recipe of the model card:

1. **Words.** The text is split at whitespace; each word keeps its character offsets.
2. **Windows.** Texts longer than the model's 512-token context are tokenised into overlapping windows
   (stride 128). All windows are scored in one batch.
3. **Voting.** Every subword of a word, in every window that contains it, adds its softmax distribution to the
   word's vote. The word takes the label with the largest mass; `conf` is that mass divided by the total.
4. **Spans.** Consecutive words with the same section form a span (the B-/I- prefixes only mark continuation).
5. **Smoothing.** Spans shorter than `min_span_words` (default 5) are merged into the preceding span, then
   neighbours with the same section are merged. This removes one-word fragments at section boundaries, but
   it also swallows genuinely short sections such as a one-word *amen* (`APPRECATIO`) or a three-word
   *Vns ist furgelegt* (`PUBLICATIO`). Use `--min-span 0` or `predict(text, min_span_words=0)` when short
   sections matter.

## Limitations

- **German `NARRATIO`.** On Middle High German charters the model tends to label the narration as `DISPOSITIO`
  (the Strasbourg mandate in the demo shows this). The model card documents this as a capacity limit of the
  117 M student, not a regression relative to the teacher.
- **Rare sections.** On the 107-charter full-scheme evaluation of the model card `SANCTIO` was not recovered at
  all and `NARRATIO` had the weakest F1 (0.80); the other sections are above 0.93.
- **Weak supervision.** The teacher's labels come from CEI annotation of a limited corpus; the student inherits
  the teacher's errors. Treat the output as a first structuring to be checked, not as an edition.
- **Genre and period.** Training data are royal and ecclesiastical charters of the 10th–14th centuries. Other
  genres (the Barletta notarial instrument in the demo), periods and orthographies can give lower accuracy.
- **Licence.** CC BY-NC 4.0: non-commercial use only.

## Demo texts

The Otto diploma is the plain text of `FreisBm/0965_IV_03.cei.xml` from the
[Structurally Annotated Medieval Charters](https://github.com/AlinaOs/Structurally_Annotated_Medieval_Charters)
corpus (Ostrowski 2021), reproduced verbatim including the transcription errors and editorial `(!)` marks of
the Monasterium.net file. This corpus is part of the teacher's training data, so the diploma illustrates the
label scheme on an in-sample text. The other three charters are not in the training corpora of the teacher or
the student.

## Citation

```bibtex
@misc{charter-diplomatic-segmentation,
  title     = {Distilled Weakly-Supervised Multi-Head Segmentation of Medieval Charters},
  author    = {Kov{\'a}cs, Tam{\'a}s and Nicolaou, Anguelos and Atzenhofer-Baumgartner, Florian and Renet, Nicolas
               and Consolo, Giuseppe and Tscherne, Niklas and Decker, Franziska and Vogeler, Georg},
  year      = {2026},
  publisher = {Hugging Face},
  doi       = {10.57967/hf/10291},
  url       = {https://huggingface.co/ERCDiDip/charter-diplomatic-segmentation}
}

@incollection{ostrowski2021formular,
  title     = {Automatische Erkennung und Klassifikation von Formularbestandteilen in K{\"o}nigsurkunden.
               Zur Aufbereitung digitaler Urkundenkorpora in der Medi{\"a}vistik},
  author    = {Ostrowski, Alina},
  booktitle = {Die Historischen Grundwissenschaften heute. Tradition -- Methodische Vielfalt -- Neuorientierung},
  editor    = {Doublier, {\'E}tienne and Schulz, Daniela and Trump, Dominik},
  pages     = {139--166},
  year      = {2021},
  address   = {Wien/K{\"o}ln/Weimar},
  doi       = {10.7788/9783412520663.139}
}
```

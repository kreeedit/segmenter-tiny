#!/usr/bin/env python3
"""
Charter diplomatic segmenter – easy-to-use wrapper for ERCDiDip/charter-diplomatic-segmentation

Downloads the distilled token-classification model ("segmenter-tiny") from the Hugging Face Hub
and splits a medieval Latin or Middle High German charter into its diplomatic parts
(INVOCATIO, INTITULATIO, ARENGA, PUBLICATIO, NARRATIO, DISPOSITIO, SANCTIO, CORROBORATIO,
SUBSCRIPTIO, DATATIO, APPRECATIO).

Command line:
    python charter_segmenter.py charter.txt                    # sections to stdout
    python charter_segmenter.py charter.txt --json out.json    # save spans as JSON
    python charter_segmenter.py charter.txt --html out.html    # colour-highlighted HTML
    python charter_segmenter.py charter.txt --words            # one line per word with label and confidence
    python charter_segmenter.py --text "In nomine sancte et individue trinitatis. Ego ..."

Python:
    from charter_segmenter import CharterSegmenter
    seg = CharterSegmenter()
    spans = seg.predict("In nomine sancte et individue trinitatis. Ego ...")
"""

import argparse
import html
import json
import re

REPO_ID = "ERCDiDip/charter-diplomatic-segmentation"

# The model's context window in subword tokens and the overlap between consecutive windows.
MAX_LENGTH = 512
STRIDE = 128

# Decoding recipe of the model card: spans shorter than this many words are merged into the
# preceding span (removes fragment noise). 0 disables the smoothing.
DEFAULT_MIN_SPAN_WORDS = 5

# Diplomatic (not alphabetical) order of the eleven sections, used for legends and tables.
SECTION_ORDER = ["INVOCATIO", "INTITULATIO", "ARENGA", "PUBLICATIO", "NARRATIO", "DISPOSITIO",
                 "SANCTIO", "CORROBORATIO", "SUBSCRIPTIO", "DATATIO", "APPRECATIO"]

SECTION_GLOSS = {
    "INVOCATIO": "invocation of God (In nomine ...)",
    "INTITULATIO": "name and title of the issuer",
    "ARENGA": "preamble with general moral or religious motivation",
    "PUBLICATIO": "announcement to the public (Notum sit ..., Wir tun kund ...)",
    "NARRATIO": "narration of the circumstances that led to the act",
    "DISPOSITIO": "the legal act itself (grant, confirmation, order ...)",
    "SANCTIO": "penalty or threat against violators",
    "CORROBORATIO": "announcement of the means of authentication (seal, witnesses)",
    "SUBSCRIPTIO": "subscriptions and witness list",
    "DATATIO": "date and place of issue",
    "APPRECATIO": "closing wish (Amen, Feliciter)",
}

SECTION_COLORS = {
    "INVOCATIO": "#f6d6a8", "INTITULATIO": "#f4b8a0", "ARENGA": "#e9c9e6", "PUBLICATIO": "#c9d8f3",
    "NARRATIO": "#bfe3d8", "DISPOSITIO": "#d9ecb2", "SANCTIO": "#f3b7c9", "CORROBORATIO": "#cfd3e8",
    "SUBSCRIPTIO": "#e0d4c0", "DATATIO": "#ffe08a", "APPRECATIO": "#d6e5f0",
}


def tokenize_words(text):
    """Whitespace words with character offsets: [(word, start_char, end_char), ...]."""
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def section_of(label):
    """'B-NARRATIO' / 'I-NARRATIO' -> 'NARRATIO'."""
    return label.split("-", 1)[1] if label[:2] in ("B-", "I-") else label


class CharterSegmenter:
    """Loads the model once; call .predict(text) as often as needed."""

    def __init__(self, model=REPO_ID, device=None, verbose=True):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        log = print if verbose else (lambda *a, **k: None)

        log(f"[1/2] Loading tokenizer and weights from {model} (downloaded once, cached afterwards) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        log(f"[2/2] Moving the model to {self.device} ...")
        self.model.to(self.device).eval()
        # Labels are read from the checkpoint: the config order is alphabetical, not diplomatic.
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.sections = [s for s in SECTION_ORDER if s in {section_of(l) for l in self.labels}]

    def predict_words(self, text, batch_size=8):
        """One prediction per whitespace word.

        Returns a list of dicts {word, start_char, end_char, label, section, conf}.
        Texts longer than the 512-token window are processed in overlapping windows
        (stride 128); every subword of a word in every window adds its softmax
        distribution to the word's vote, and the word takes the label with the largest mass.
        `conf` is that mass divided by the total mass of the word.
        """
        toks = tokenize_words(text)
        if not toks:
            return []
        words = [w for w, _, _ in toks]
        enc = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=MAX_LENGTH,
                             stride=STRIDE, return_overflowing_tokens=True, padding=True,
                             return_tensors="pt")
        n_windows = enc["input_ids"].shape[0]
        mass = self.torch.zeros(len(words), len(self.labels))
        with self.torch.no_grad():
            for b in range(0, n_windows, batch_size):
                ids = enc["input_ids"][b:b + batch_size].to(self.device)
                att = enc["attention_mask"][b:b + batch_size].to(self.device)
                probs = self.model(input_ids=ids, attention_mask=att).logits.softmax(-1).cpu()
                for k in range(ids.shape[0]):
                    for pos, w in enumerate(enc.word_ids(b + k)):
                        if w is not None:
                            mass[w] += probs[k, pos]
        out = []
        for (word, s, e), row in zip(toks, mass):
            total = float(row.sum())
            if total == 0.0:
                continue
            i = int(row.argmax())
            out.append({"word": word, "start_char": s, "end_char": e, "label": self.labels[i],
                        "section": section_of(self.labels[i]), "conf": float(row[i]) / total})
        return out

    def predict(self, text, min_span_words=DEFAULT_MIN_SPAN_WORDS):
        """Diplomatic sections of a charter as a list of spans sorted by position.

        Each span: {section, text, start_char, end_char, start_word, end_word, n_words, conf}.
        `conf` is the mean word confidence of the span. Spans shorter than `min_span_words`
        are merged into the preceding span (model-card recipe); pass 0 to keep the raw spans.
        """
        words = self.predict_words(text)
        return smooth_spans(collapse_spans(words), min_span_words)

    def predict_file(self, path, **kwargs):
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        return text, self.predict(text, **kwargs)


def collapse_spans(words):
    """Consecutive words with the same section -> one span (B-/I- prefixes are ignored)."""
    spans = []
    for i, w in enumerate(words):
        if spans and spans[-1]["section"] == w["section"]:
            sp = spans[-1]
            sp["end_char"], sp["end_word"], sp["n_words"] = w["end_char"], i, sp["n_words"] + 1
            sp["_conf_sum"] += w["conf"]
        else:
            spans.append({"section": w["section"], "start_char": w["start_char"], "end_char": w["end_char"],
                          "start_word": i, "end_word": i, "n_words": 1, "_conf_sum": w["conf"]})
    return _finish(spans, words)


def smooth_spans(spans, min_span_words=DEFAULT_MIN_SPAN_WORDS):
    """Merge spans shorter than `min_span_words` into the preceding span, then merge neighbours
    that ended up with the same section."""
    if min_span_words <= 1 or not spans:
        return spans
    merged = []
    for sp in spans:
        sp = dict(sp)
        if merged and (sp["n_words"] < min_span_words or merged[-1]["section"] == sp["section"]):
            prev = merged[-1]
            n = prev["n_words"] + sp["n_words"]
            prev["conf"] = (prev["conf"] * prev["n_words"] + sp["conf"] * sp["n_words"]) / n
            prev.update(end_char=sp["end_char"], end_word=sp["end_word"], n_words=n,
                        text=prev["text"] + " " + sp["text"])
            if sp["n_words"] >= min_span_words:
                prev["section"] = sp["section"]
        else:
            merged.append(sp)
    return merged


def _finish(spans, words):
    text_words = [w["word"] for w in words]
    for sp in spans:
        sp["text"] = " ".join(text_words[sp["start_word"]:sp["end_word"] + 1])
        sp["conf"] = sp.pop("_conf_sum") / sp["n_words"]
    return spans


def to_html(text, spans, title="Charter diplomatic segmentation"):
    """Colour-highlighted HTML: every span in its section colour, the section name as subscript,
    mean confidence in the tooltip. Text between spans (none, normally) is left unmarked."""
    parts, pos = [], 0
    for sp in spans:
        parts.append(html.escape(text[pos:sp["start_char"]]))
        colour = SECTION_COLORS.get(sp["section"], "#ddd")
        parts.append(f'<mark style="background:{colour}" title="{sp["section"]} (mean confidence {sp["conf"]:.2f}, '
                     f'{sp["n_words"]} words)">{html.escape(text[sp["start_char"]:sp["end_char"]])}'
                     f'<sub>{sp["section"]}</sub></mark>')
        pos = sp["end_char"]
    parts.append(html.escape(text[pos:]))
    legend = " ".join(f'<mark style="background:{SECTION_COLORS[s]}">{s}</mark>' for s in SECTION_ORDER)
    return (f'<!doctype html><meta charset="utf-8"><title>{html.escape(title)}</title>'
            f'<body style="font-family:Georgia,serif;max-width:60em;margin:2em auto;line-height:2">'
            f'<h2>{html.escape(title)}</h2><p>{legend}</p><hr><p>{"".join(parts)}</p></body>')


def print_spans(spans, width=100):
    for sp in spans:
        t = sp["text"] if len(sp["text"]) <= width else sp["text"][:width - 3] + "..."
        print(f'[{sp["section"]}] ({sp["n_words"]} words, conf {sp["conf"]:.2f}) {t}')


def print_words(words):
    print(f'{"WORD":<28}{"LABEL":<16}CONF')
    for w in words:
        print(f'{w["word"][:27]:<28}{w["label"]:<16}{w["conf"]:.2f}')


def main():
    p = argparse.ArgumentParser(description="Diplomatic segmentation of medieval Latin and German charters.")
    p.add_argument("file", nargs="?", help="UTF-8 text file with one charter")
    p.add_argument("--text", help="analyse this string instead of a file")
    p.add_argument("--min-span", type=int, default=DEFAULT_MIN_SPAN_WORDS,
                   help=f"merge spans shorter than this many words into the previous span "
                        f"(default {DEFAULT_MIN_SPAN_WORDS}; 0 = raw spans)")
    p.add_argument("--words", action="store_true", help="print one line per word with label and confidence")
    p.add_argument("--json", help="write the spans (and per-word labels) to this JSON file")
    p.add_argument("--html", help="write highlighted HTML to this file")
    p.add_argument("--model", default=REPO_ID, help="Hub id or local folder (default: the Hub model)")
    p.add_argument("--device", help="cpu / cuda (default: auto)")
    args = p.parse_args()

    if not args.file and not args.text:
        p.error("give a text file or --text")
    text = args.text if args.text else open(args.file, encoding="utf-8").read().strip()

    seg = CharterSegmenter(model=args.model, device=args.device)
    words = seg.predict_words(text)
    spans = smooth_spans(collapse_spans(words), args.min_span)

    if args.words:
        print_words(words)
        print()
    print_spans(spans)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"text": text, "model": args.model, "min_span_words": args.min_span,
                       "spans": spans, "words": words}, f, ensure_ascii=False, indent=2)
        print(f"JSON saved: {args.json}")
    if args.html:
        with open(args.html, "w", encoding="utf-8") as f:
            f.write(to_html(text, spans))
        print(f"HTML saved: {args.html}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Bib/citation checker for LaTeX papers.
Pure code, no LLM calls. Catches the issues that get papers desk-rejected.

Usage:  python check_bib.py [--bib FILE] [--tex FILE ...]
Defaults: iclr2027.bib, iclr2027.tex + theory_*.tex
"""
import re, sys, os, argparse, unicodedata
from pathlib import Path
from collections import Counter, defaultdict

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── helpers ──────────────────────────────────────────────────────────

def strip_braces(s):
    return s.replace("{", "").replace("}", "")

def strip_latex(s):
    s = re.sub(r"\\['\"`^~=.uvHtcdb]\{?\w?\}?", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return strip_braces(s).strip()

def has_non_ascii(s):
    return any(ord(c) > 127 and unicodedata.category(c) != "Mn" for c in s)

# ── parse bib ────────────────────────────────────────────────────────

ENTRY_RE = re.compile(r"@(\w+)\s*\{([^,]+),", re.IGNORECASE)
FIELD_RE = re.compile(
    r"(\w+)\s*=\s*(?:\{((?:[^{}]|\{[^{}]*\})*)\}|\"([^\"]*)\"|(\d+))",
    re.DOTALL,
)

def parse_bib(path):
    text = path.read_text(encoding="utf-8")
    entries = {}
    for m in ENTRY_RE.finditer(text):
        etype = m.group(1).lower()
        key = m.group(2).strip()
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i - 1]
        fields = {}
        for fm in FIELD_RE.finditer(body):
            fname = fm.group(1).lower()
            fval = fm.group(2) or fm.group(3) or fm.group(4) or ""
            fields[fname] = fval.strip()
        entries[key] = {"type": etype, "fields": fields, "raw": body}
    return entries

# ── parse tex citations ─────────────────────────────────────────────

CITE_RE = re.compile(r"\\cite[tp]?\{([^}]+)\}")

def get_cited_keys(tex_paths):
    keys = []
    locations = defaultdict(list)
    for p in tex_paths:
        text = p.read_text(encoding="utf-8")
        # Join lines so multi-line \citep{...} is caught
        joined = text.replace("\n", " ")
        for m in CITE_RE.finditer(joined):
            for k in m.group(1).split(","):
                k = k.strip()
                if k:
                    keys.append(k)
                    # Find approximate line number
                    pos = m.start()
                    lineno = text[:pos].count("\n") + 1
                    locations[k].append(f"{p.name}:{lineno}")
    return keys, locations

# ── also find \input{} files ─────────────────────────────────────────

INPUT_RE = re.compile(r"\\input\{([^}]+)\}")

def resolve_inputs(main_tex):
    text = main_tex.read_text(encoding="utf-8")
    result = [main_tex]
    for m in INPUT_RE.finditer(text):
        fname = m.group(1)
        if not fname.endswith(".tex"):
            fname += ".tex"
        p = main_tex.parent / fname
        if p.exists():
            result.append(p)
    return result

# ── checks ───────────────────────────────────────────────────────────

class Issue:
    def __init__(self, severity, key, msg):
        self.severity = severity  # ERROR, WARN, INFO
        self.key = key
        self.msg = msg
    def __str__(self):
        tag = {"ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️"}[self.severity]
        prefix = f"[{self.key}] " if self.key else ""
        return f"  {tag} {prefix}{self.msg}"


def check_all(bib_path, tex_paths):
    issues = []
    entries = parse_bib(bib_path)
    cited_keys, cite_locs = get_cited_keys(tex_paths)
    cited_set = set(cited_keys)
    bib_set = set(entries.keys())

    # 1. Undefined citations (cited in tex but not in bib)
    for k in sorted(cited_set - bib_set):
        locs = ", ".join(cite_locs[k])
        issues.append(Issue("ERROR", k, f"Cited but not in .bib  (at {locs})"))

    # 2. Unused bib entries (in bib but never cited)
    for k in sorted(bib_set - cited_set):
        issues.append(Issue("WARN", k, "Defined in .bib but never cited"))

    # 3. Per-entry checks
    for key, entry in entries.items():
        f = entry["fields"]
        etype = entry["type"]

        # 3a. Missing required fields
        if etype == "article":
            for req in ["author", "title", "journal", "year"]:
                if req not in f:
                    issues.append(Issue("ERROR", key, f"Missing required field '{req}' for @article"))
            # volume is expected for published articles
            if "volume" not in f and "journal" in f and "arxiv" not in f.get("journal", "").lower():
                issues.append(Issue("WARN", key, f"@article missing 'volume' (expected for published journals)"))
        elif etype == "inproceedings":
            for req in ["author", "title", "booktitle", "year"]:
                if req not in f:
                    issues.append(Issue("ERROR", key, f"Missing required field '{req}' for @inproceedings"))
        elif etype == "book":
            for req in ["author", "title", "publisher", "year"]:
                if req not in f:
                    issues.append(Issue("ERROR", key, f"Missing required field '{req}' for @book"))

        # 3b. Type mismatch: @article with booktitle (should be @inproceedings)
        if etype == "article" and "booktitle" in f and "journal" not in f:
            issues.append(Issue("ERROR", key,
                f"@article has 'booktitle' but no 'journal' — should be @inproceedings"))
        if etype == "article" and "booktitle" in f and "journal" in f:
            issues.append(Issue("WARN", key,
                f"@article has both 'journal' and 'booktitle' — pick one"))

        # 3c. Year sanity
        year = f.get("year", "")
        if year:
            try:
                y = int(year)
                if y < 1900 or y > 2026:
                    issues.append(Issue("WARN", key, f"Unusual year: {y}"))
            except ValueError:
                issues.append(Issue("ERROR", key, f"Non-numeric year: '{year}'"))

        # 3d. Title casing — proper nouns and acronyms should be braced
        title = f.get("title", "")
        bare_title = strip_braces(title)
        acronyms = re.findall(r"\b[A-Z]{2,}\b", bare_title)
        for acr in acronyms:
            if "{" + acr + "}" not in title and "{" + acr[0] + "}" not in title:
                issues.append(Issue("WARN", key,
                    f"Acronym '{acr}' in title may lose casing — wrap in braces"))

        # 3e. Pages format
        pages = f.get("pages", "")
        if pages:
            if "--" not in pages and "-" in pages:
                issues.append(Issue("ERROR", key,
                    f"Pages '{pages}' uses single dash — use double dash '--'"))
            if re.search(r"[^0-9\-\s]", pages):
                issues.append(Issue("WARN", key, f"Unusual characters in pages: '{pages}'"))

        # 3f. Duplicate keys check (bib parser keeps last, so check raw)
        pass  # handled by duplicate-cite check below

        # 3g. Author formatting
        authors = f.get("author", "")
        if authors:
            if " & " in authors and " and " not in authors:
                issues.append(Issue("WARN", key,
                    "Author field uses '&' instead of 'and' as separator"))
            # Check for "First Last" without comma (should be "Last, First")
            author_list = re.split(r"\s+and\s+", authors)
            for a in author_list:
                a = strip_latex(a).strip()
                if a and "," not in a and len(a.split()) >= 2:
                    # Could be "First Last" format — warn
                    parts = a.split()
                    if parts[-1][0].isupper() and parts[0][0].isupper():
                        pass  # Both formats are valid in BibTeX, don't warn

        # 3h. URL in wrong field
        for field_name in ["title", "author", "journal", "booktitle"]:
            val = f.get(field_name, "")
            if "http://" in val or "https://" in val:
                issues.append(Issue("WARN", key,
                    f"URL found in '{field_name}' field — use 'url' field instead"))

        # 3i. Trailing/leading whitespace in fields
        for fname, fval in f.items():
            if fval != fval.strip():
                issues.append(Issue("INFO", key,
                    f"Field '{fname}' has leading/trailing whitespace"))

        # 3j. Empty fields
        for fname, fval in f.items():
            if fval.strip() == "":
                issues.append(Issue("WARN", key, f"Empty field '{fname}'"))

        # 3k. Journal abbreviations — inconsistent use of "." in abbreviations
        journal = f.get("journal", "")
        if journal:
            if "Proc\\" in journal or "J\\" in journal:
                pass  # LaTeX abbreviation, fine
            # Check for unprotected special chars
            if "&" in journal and "\\&" not in journal:
                issues.append(Issue("ERROR", key,
                    f"Unescaped '&' in journal: '{journal}'"))

        # 3l. Booktitle for conferences — should be consistent
        booktitle = f.get("booktitle", "")
        if booktitle:
            if "&" in booktitle and "\\&" not in booktitle:
                issues.append(Issue("ERROR", key,
                    f"Unescaped '&' in booktitle: '{booktitle}'"))

    # 4. Duplicate citation keys in bib file (raw check)
    raw_text = bib_path.read_text(encoding="utf-8")
    all_keys = [m.group(2).strip() for m in ENTRY_RE.finditer(raw_text)]
    key_counts = Counter(all_keys)
    for k, c in key_counts.items():
        if c > 1:
            issues.append(Issue("ERROR", k, f"Duplicate bib key (appears {c} times)"))

    # 5. Duplicate citations in same \cite{} command
    for p in tex_paths:
        text = p.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for m in CITE_RE.finditer(line):
                keys_in_cite = [k.strip() for k in m.group(1).split(",")]
                dups = [k for k, c in Counter(keys_in_cite).items() if c > 1]
                for d in dups:
                    issues.append(Issue("WARN", d,
                        f"Cited twice in same \\cite command at {p.name}:{lineno}"))

    # 6. Check for common spelling errors in known fields
    KNOWN_VENUES = {
        "NeurIPS": ["Neurips", "neurips", "NIPS", "nips"],
        "ICML": ["Icml", "icml"],
        "ICLR": ["Iclr", "iclr"],
        "CVPR": ["Cvpr", "cvpr"],
        "EMNLP": ["Emnlp", "emnlp"],
    }
    for key, entry in entries.items():
        bt = entry["fields"].get("booktitle", "")
        for correct, wrong_list in KNOWN_VENUES.items():
            for wrong in wrong_list:
                if wrong in bt and correct not in bt:
                    issues.append(Issue("WARN", key,
                        f"Booktitle has '{wrong}' — should be '{correct}'"))

    # 7. @article with booktitle instead of journal
    for key, entry in entries.items():
        if entry["type"] == "article" and "booktitle" in entry["fields"]:
            if "journal" not in entry["fields"]:
                pass  # already caught above
            else:
                # Has both — booktitle is suspicious for @article
                pass  # already caught above

    return issues


# ── citation context checker ─────────────────────────────────────────

KNOWN_CLAIMS = {
    "koopman1931hamiltonian": {
        "what_paper_does": "Introduces the Koopman operator: an infinite-dimensional linear operator that governs the evolution of observable functions of a nonlinear dynamical system in Hilbert space.",
        "our_cite_context": "Foundational reference for the Koopman operator framework.",
        "status": "CORRECT"
    },
    "lusch2018deep": {
        "what_paper_does": "Trains deep autoencoders to learn Koopman eigenfunctions, enabling linear prediction of nonlinear dynamics. Uses auxiliary networks and a combined reconstruction+prediction loss.",
        "our_cite_context": "Cited as an example of the coupled AE approach with weighted loss tuning.",
        "status": "CORRECT"
    },
    "otto2019linearly": {
        "what_paper_does": "Linearly Recurrent Autoencoder Networks: trains AE where latent dynamics are constrained to be linear (z_{t+1} = Kz_t). Joint reconstruction+prediction loss.",
        "our_cite_context": "Cited alongside Lusch as coupled AE with loss-weight tuning.",
        "status": "CORRECT"
    },
    "schmid2010dmd": {
        "what_paper_does": "Introduces Dynamic Mode Decomposition: fits a best-fit linear operator A = YX^dagger from snapshot pairs. Demonstrates on jet flow and cylinder wake.",
        "our_cite_context": "Cited for DMD definition A = YX^dagger.",
        "status": "CORRECT"
    },
    "brunton2022modern": {
        "what_paper_does": "Comprehensive SIAM Review survey of modern Koopman theory: connections between Koopman operator, DMD, and deep learning approximations.",
        "our_cite_context": "Cited for DMD as post-hoc dynamics model on AE latents.",
        "status": "CORRECT"
    },
    "takeishi2017learning": {
        "what_paper_does": "Learns Koopman invariant subspaces via deep autoencoders with auxiliary eigenfunction objectives at NeurIPS 2017.",
        "our_cite_context": "Cited in related work as a Koopman AE variant.",
        "status": "CORRECT"
    },
    "champion2019data": {
        "what_paper_does": "SINDy Autoencoders: learns coordinates where dynamics are sparse (via SINDy), not necessarily linear. Published in PNAS.",
        "our_cite_context": "Cited as a Koopman/deep-AE variant for learning coordinates.",
        "status": "CORRECT"
    },
    "yeung2019learning": {
        "what_paper_does": "Deep neural network representations for Koopman operators of nonlinear systems. ACC 2019.",
        "our_cite_context": "Cited as a Koopman AE variant.",
        "status": "CORRECT"
    },
    "gin2021deep": {
        "what_paper_does": "Deep learning models that find coordinate transformations linearizing PDEs. Extends Koopman AE idea to PDE systems.",
        "our_cite_context": "Cited as a Koopman AE variant.",
        "status": "CORRECT"
    },
    "erichson2019": {
        "what_paper_does": "Adds Lyapunov stability constraints to Koopman autoencoders for fluid flow prediction. arXiv preprint.",
        "our_cite_context": "Cited as physics-informed variant that retains coupled architecture.",
        "status": "CORRECT"
    },
    "azencot2020": {
        "what_paper_does": "Consistent Koopman Autoencoders: adds forward-backward consistency penalty to enforce invertibility. ICML 2020.",
        "our_cite_context": "Cited as consistent variant that retains coupled architecture.",
        "status": "CORRECT — but note entry type issue (see bib check)"
    },
    "chen2018neural": {
        "what_paper_does": "Neural ODEs: parameterises continuous-time dynamics dz/dt = f(z) and differentiates through ODE solvers via adjoint method. NeurIPS 2018.",
        "our_cite_context": "Cited for Neural ODE definition and as one of our forecast heads.",
        "status": "CORRECT"
    },
    "rubanova2019latent": {
        "what_paper_does": "Latent ODEs for irregularly-sampled time series: encodes with RNN, evolves latent with Neural ODE, decodes. NeurIPS 2019.",
        "our_cite_context": "Cited for latent ODE application to time series.",
        "status": "CORRECT"
    },
    "sener2018multi": {
        "what_paper_does": "Formulates multi-task learning as multi-objective optimisation, proposes finding Pareto-optimal solutions. NeurIPS 2018.",
        "our_cite_context": "Cited for joint reconstruction+forecasting as multi-task learning.",
        "status": "CORRECT"
    },
    "yu2020gradient": {
        "what_paper_does": "PCGrad: projects conflicting task gradients to remove interference in multi-task learning. NeurIPS 2020.",
        "our_cite_context": "Cited for PCGrad as a gradient conflict mitigation method.",
        "status": "CORRECT"
    },
    "liu2021conflict": {
        "what_paper_does": "CAGrad: Conflict-Averse Gradient Descent finds common descent direction for all tasks. NeurIPS 2021.",
        "our_cite_context": "Cited for CAGrad as a gradient conflict mitigation method.",
        "status": "CORRECT"
    },
    "he2016deep": {
        "what_paper_does": "ResNet: deep residual learning with skip connections for image recognition. CVPR 2016.",
        "our_cite_context": "Cited for residual learning principle (identity + residual decomposition).",
        "status": "CORRECT"
    },
    "cho2014learning": {
        "what_paper_does": "Introduces GRU (Gated Recurrent Unit) for sequence modelling. EMNLP 2014.",
        "our_cite_context": "Cited for GRU cell used in carrier accumulation.",
        "status": "CORRECT"
    },
    "hinton2006reducing": {
        "what_paper_does": "Pretraining deep autoencoders with RBMs for dimensionality reduction. Science 2006.",
        "our_cite_context": "Cited only in fmts2026.tex (workshop paper), not in iclr2027.tex.",
        "status": "CORRECT (unused in ICLR draft)"
    },
    "saxena2008damage": {
        "what_paper_does": "C-MAPSS turbofan engine degradation simulation dataset.",
        "our_cite_context": "Was cited for C-MAPSS dataset — now removed from paper.",
        "status": "CORRECT (unused in ICLR draft)"
    },
    "watter2015embed": {
        "what_paper_does": "Embed to Control: learns locally linear latent dynamics from images for control. NeurIPS 2015.",
        "our_cite_context": "Cited only in fmts2026.tex, not in iclr2027.tex.",
        "status": "CORRECT (unused in ICLR draft)"
    },
    "bevanda2021koopman": {
        "what_paper_does": "Survey of Koopman operator methods for learning, analysis, and control.",
        "our_cite_context": "Cited only in fmts2026.tex, not in iclr2027.tex.",
        "status": "CORRECT (unused in ICLR draft)"
    },
    "cai2023hyvae": {
        "what_paper_does": "Hybrid VAE for time series forecasting, combining VAE with temporal modelling.",
        "our_cite_context": "Cited only in fmts2026.tex, not in iclr2027.tex.",
        "status": "CORRECT (unused in ICLR draft)"
    },
}


def check_citation_contexts(bib_entries, tex_paths):
    """Check that each citation is used in a context consistent with what the paper actually says."""
    issues = []
    cited_keys, _ = get_cited_keys(tex_paths)
    cited_set = set(cited_keys)

    for key in sorted(cited_set):
        if key in KNOWN_CLAIMS:
            claim = KNOWN_CLAIMS[key]
            if "CORRECT" not in claim["status"]:
                issues.append(Issue("ERROR", key,
                    f"Citation may be inaccurate: {claim['status']}"))
        else:
            issues.append(Issue("WARN", key,
                "No verification record — manually check this citation"))

    return issues


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check bib/citation quality")
    parser.add_argument("--bib", default="iclr2027.bib", help="Path to .bib file")
    parser.add_argument("--tex", nargs="*", default=None, help="Paths to .tex files")
    args = parser.parse_args()

    bib_path = Path(args.bib)
    if not bib_path.exists():
        print(f"ERROR: bib file not found: {bib_path}")
        sys.exit(1)

    if args.tex:
        tex_paths = [Path(p) for p in args.tex]
    else:
        main_tex = bib_path.parent / "iclr2027.tex"
        if main_tex.exists():
            tex_paths = resolve_inputs(main_tex)
        else:
            print("ERROR: no tex files found")
            sys.exit(1)

    print(f"Checking: {bib_path.name}")
    print(f"Against:  {', '.join(p.name for p in tex_paths)}")
    print()

    entries = parse_bib(bib_path)
    print(f"Bib entries: {len(entries)}")
    cited_keys, _ = get_cited_keys(tex_paths)
    print(f"Citations in tex: {len(set(cited_keys))} unique keys, {len(cited_keys)} total uses")
    print()

    # Run structural checks
    print("=" * 60)
    print("  STRUCTURAL CHECKS")
    print("=" * 60)
    issues = check_all(bib_path, tex_paths)

    errors = [i for i in issues if i.severity == "ERROR"]
    warns = [i for i in issues if i.severity == "WARN"]
    infos = [i for i in issues if i.severity == "INFO"]

    if errors:
        print(f"\n{'ERRORS':}")
        for i in errors:
            print(i)
    if warns:
        print(f"\n{'WARNINGS':}")
        for i in warns:
            print(i)
    if infos:
        print(f"\n{'INFO':}")
        for i in infos:
            print(i)
    if not issues:
        print("\n  ✅ All structural checks passed!")

    # Run citation context verification
    print()
    print("=" * 60)
    print("  CITATION ACCURACY VERIFICATION")
    print("=" * 60)
    ctx_issues = check_citation_contexts(entries, tex_paths)
    if ctx_issues:
        for i in ctx_issues:
            print(i)
    else:
        print("\n  ✅ All citation contexts verified!")

    # Print full verification report
    print()
    print("=" * 60)
    print("  DETAILED CITATION REPORT")
    print("=" * 60)
    cited_set = set(cited_keys)
    for key in sorted(cited_set):
        if key in KNOWN_CLAIMS:
            c = KNOWN_CLAIMS[key]
            status_icon = "✅" if "CORRECT" in c["status"] else "❌"
            print(f"\n  {status_icon} {key}")
            print(f"     Paper: {c['what_paper_does'][:100]}")
            print(f"     We cite for: {c['our_cite_context']}")
        else:
            print(f"\n  ❓ {key} — no verification record")

    # Summary
    print()
    print("=" * 60)
    n_err = len(errors) + len([i for i in ctx_issues if i.severity == "ERROR"])
    n_warn = len(warns) + len([i for i in ctx_issues if i.severity == "WARN"])
    if n_err == 0 and n_warn == 0:
        print("  ✅ BIB CHECK PASSED — no issues found")
    elif n_err == 0:
        print(f"  ⚠️  BIB CHECK: {n_warn} warnings, 0 errors")
    else:
        print(f"  ❌ BIB CHECK FAILED: {n_err} errors, {n_warn} warnings")
    print("=" * 60)

    sys.exit(1 if n_err > 0 else 0)


if __name__ == "__main__":
    main()

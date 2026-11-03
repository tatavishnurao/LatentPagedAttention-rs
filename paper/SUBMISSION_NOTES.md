# ISPASS 2027 submission notes (decisions for the author)

Rules below were read on 2026-11-03 from https://ispass.org/2027/cfp/ and
https://ispass.org/2027/submission/ (the submission site was not open yet). Nothing in this file is
in the PDF. Laptop RTX 4060, WSL2, not a cloud result.

## Venue rules as read

- Nine double-column pages, 10-point type, US Letter, IEEE conference style; unlimited pages of
  references; everything else (figures, appendices) within the first nine.
- Double-blind; cite own work in the third person; do not omit references for anonymity.
- "References must list all authors."
- "Please number the pages of the submission." (`build_ieee.py` adds `\pagestyle{plain}`.)
- Generative AI: editing and polishing need no disclosure; "Any substantial use, for example to
  generate text, code, data, figures or citations, must be disclosed in the paper's
  acknowledgments." AI tools may not be authors.
- arXiv: allowed before or after submission (notify the Program Chair if after); different titles
  recommended for the two versions; no social-media advertising during review.
- Abstract due 2026-12-11, full paper 2026-12-18 (AoE); accepted papers must be presented in person
  in New York City, 2027-05-09 to 11.

## Build status

`paper/ieee/paper.pdf`: 8 pages, references from page 7, so everything except references fits in
under 7 of the 9 allowed pages. IEEEtran conference class unmodified (packages for encoding, Unicode,
graphics, table rules and URLs only); page numbers on. Template caveat: the CFP points to the IEEE
templates page; `IEEEtran.cls` from the Ubuntu `texlive-publishers` package was used, not a file downloaded from that page.

## 1. AI-use disclosure (draft, needs your approval before it goes into the acknowledgments)

> Acknowledgments. AI assistants (Anthropic's Claude) were used to write and test code and
> measurement harnesses, to draft and revise the text, and to produce figures. All measurements
> come from the authors' own runs on the hardware described, every number in the paper is generated
> from committed result files, and the authors checked every citation against its source.

Check before approving: the product name, whether "every citation against its source" is true for
you (the agents read each source; you have not re-read them), and whether you want the
acknowledgments to also thank anyone else.

## 2. arXiv title (proposal; the CFP recommends a different title)

Proposed: **"Fewer Bytes, Same Time: Paged Latent-Cache Decode and Clock-Validity Traps on a
Laptop GPU"**

Submission title stays "Compression Is Not Speedup: Measuring Paged Latent-Cache Decode Attention
on a Consumer GPU".

## 3. Decisions still needed

- **Reference [2], DeepSeek-V2.** Cited as the corporate author "DeepSeek-AI", exactly as its arXiv
  page lists it ("DeepSeek-AI and 156 other authors"). The CFP says "References must list all
  authors"; 157 names would need reading from the paper itself, and would take several column
  inches. Your call: corporate author only, or the full list.
- **Wide tables.** Tables 2, 3 and 5 span both columns in addition to Table 4, because in one column
  they overflowed by 72-111 pt. Alternative: shorter cells or splitting them.
- **Author line, affiliation, email** for the camera-ready (the submission stays anonymous).
- **Abstract registration** by 2026-12-11.

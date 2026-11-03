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

## Decisions taken (2026-11-03)

- **AI-use acknowledgment:** approved with the last clause changed to "every citation was read at
  its source before being cited"; it is in the paper as an unnumbered Acknowledgment section.
- **arXiv title:** "Fewer Bytes, Same Time: Paged Latent-Cache Decode and Clock-Validity Traps on a
  Laptop GPU". The submission title stays "Compression Is Not Speedup: Measuring Paged Latent-Cache
  Decode Attention on a Consumer GPU".
- **Reference [2]:** stays the corporate author "DeepSeek-AI", as its arXiv page lists it.
- **Wide tables:** Tables 2, 3 and 5 span both columns, approved.
- **Page numbers:** on, approved.
- **Venue strings:** all unverified ones removed; references give arXiv IDs and DOIs only, except
  [9] (ASPLOS '09, confirmed via its DOI record).

## Still needed

- Author line, affiliation and email for the camera-ready (the submission stays anonymous).
- Abstract registration by 2026-12-11; full paper by 2026-12-18 (AoE).

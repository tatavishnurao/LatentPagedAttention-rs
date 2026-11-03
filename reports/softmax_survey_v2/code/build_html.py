"""Render report.md and blog.md to site/report.html and site/blog.html.

* $...$ and $$...$$ -> MathML (latex2mathml); code blocks are protected first.
* ![alt](src "caption") on its own line -> <figure> with caption.
* [@key] / [@a; @b] -> numbered citation links; {{REFERENCES}} -> reference list built from
  sources/references.json + sources/arxiv_meta.json (full author lists).
* [proven] [paper] [config] [measured] [inference] -> small tag chips.
Requires: markdown, latex2mathml (see requirements.txt)."""
import html
import json
import os
import re
import markdown
from latex2mathml.converter import convert as l2m

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
os.makedirs(SITE, exist_ok=True)
REFS = json.load(open(os.path.join(ROOT, "sources", "references.json")))
META = json.load(open(os.path.join(ROOT, "sources", "arxiv_meta.json")))

CSS = """
/* Layout: one reading column (~72ch); figures may run wider; quiet technical-report styling. */
:root{
  --bg:#fbfbf9; --fg:#17191e; --muted:#596070; --rule:#dcdfe4; --accent:#1f5ea8;
  --panel:#eff2f6; --plate:#ffffff; --code-bg:#edf0f4;
  --t-proven:#1d6b46; --t-paper:#1f5ea8; --t-config:#6a4c93; --t-measured:#9a5b00; --t-inference:#8a3a3a;
  --f-body:"Source Serif 4", Georgia, "Times New Roman", serif;
  --f-head:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#14161a; --fg:#e6e7ea; --muted:#a2a8b3; --rule:#2d3139; --accent:#86b6f0;
  --panel:#1c2026; --plate:#ffffff; --code-bg:#1f232a;
  --t-proven:#6fd0a0; --t-paper:#86b6f0; --t-config:#c3a6ee; --t-measured:#f0b866; --t-inference:#f09a9a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#14161a; --fg:#e6e7ea; --muted:#a2a8b3; --rule:#2d3139; --accent:#86b6f0;
  --panel:#1c2026; --plate:#ffffff; --code-bg:#1f232a;
  --t-proven:#6fd0a0; --t-paper:#86b6f0; --t-config:#c3a6ee; --t-measured:#f0b866; --t-inference:#f09a9a; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:17px/1.62 var(--f-body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:40px 80px}
main{max-width:72ch;margin:0 auto;min-width:0}
h1,h2,h3{font-family:var(--f-head);line-height:1.2;text-wrap:balance;letter-spacing:.005em}
h1{font-size:2.1rem;margin:0 0 .3rem;font-weight:600}
h2{font-size:1.45rem;margin:2.6rem 0 .6rem;padding-top:1rem;border-top:1px solid var(--rule);font-weight:600}
h3{font-size:1.12rem;margin:1.8rem 0 .4rem;font-weight:600}
h1+p strong{font-family:var(--f-head);font-weight:500;color:var(--muted);font-size:1.15rem}
a{color:var(--accent);text-underline-offset:2px;overflow-wrap:anywhere}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
hr{border:0;border-top:1px solid var(--rule);margin:1.6rem 0}
code{font-family:var(--f-mono);font-size:.84em;background:var(--code-bg);padding:.08em .3em;border-radius:3px}
pre{background:var(--code-bg);padding:12px 14px;border-radius:6px;overflow-x:auto;font-size:.82rem;line-height:1.5}
pre code{background:none;padding:0}
.table-wrap{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;font-size:.84rem;line-height:1.4;min-width:560px;width:100%;font-variant-numeric:tabular-nums}
th,td{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;vertical-align:top}
th{font-family:var(--f-head);font-weight:600;font-size:.8rem;letter-spacing:.03em;text-transform:uppercase;color:var(--muted)}
figure{margin:1.8rem 0;width:min(100vw - 32px, 1000px);position:relative;left:50%;transform:translateX(-50%)}
figure .plate{background:var(--plate);border:1px solid var(--rule);border-radius:6px;padding:10px;overflow-x:auto}
figure img{display:block;margin:0 auto;max-width:100%;height:auto}
figcaption{max-width:72ch;margin:.7rem auto 0;font-size:.9rem;line-height:1.5;color:var(--muted)}
figcaption strong{color:var(--fg);font-family:var(--f-head);font-weight:600}
blockquote{background:var(--panel);border-radius:6px;padding:4px 18px;margin:1.4rem 0;font-size:.96rem}
math{font-family:"STIX Two Math","Cambria Math","Latin Modern Math",serif;font-size:1.05em}
.math-block{overflow-x:auto;margin:1rem 0;padding:2px 0}
.tag{font-family:var(--f-mono);font-size:.72em;letter-spacing:.02em;white-space:nowrap;border:1px solid currentColor;border-radius:3px;padding:0 .3em;margin:0 .1em}
.tag-proven{color:var(--t-proven)}.tag-paper{color:var(--t-paper)}.tag-config{color:var(--t-config)}
.tag-measured{color:var(--t-measured)}.tag-inference{color:var(--t-inference)}
.cite{font-family:var(--f-head);font-size:.86em;overflow-wrap:anywhere}
ol.refs{font-size:.88rem;line-height:1.45;padding-left:2.2em}
ol.refs li{margin:.45rem 0}
ol.refs details{margin-top:.2rem;font-size:.82rem;color:var(--muted)}
ol.refs summary{cursor:pointer;color:var(--accent)}
.vtag{font-family:var(--f-mono);font-size:.78em;color:var(--muted)}
.backlink{font-size:.9rem;color:var(--muted)}
@media (max-width:520px){body{font-size:16px} h1{font-size:1.7rem}}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
"""

HEAD = """<title>{title}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{css}</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
"""


def math(src):
    def block(m):
        try:
            return '<div class="math-block">' + l2m(m.group(1).strip(), display="block") + "</div>"
        except Exception:
            return "<pre>" + html.escape(m.group(1)) + "</pre>"

    def inline(m):
        try:
            return l2m(m.group(1))
        except Exception:
            return "<code>" + html.escape(m.group(1)) + "</code>"
    src = re.sub(r"\$\$(.+?)\$\$", block, src, flags=re.S)
    src = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", inline, src)
    return src


def cite_numbers(text):
    order = []
    for m in re.finditer(r"\[@([^\]]+)\]", text):
        for k in re.findall(r"@([\w-]+)", "@" + m.group(1)):
            if k not in order:
                order.append(k)
    return order


def render_refs(order):
    items = []
    for k in order:
        if k in REFS["arxiv"]:
            aid, note = REFS["arxiv"][k]
            m = META[aid]
            auth = [a.strip() for a in m["authors"] if a.strip() not in (":",)]
            full = ""
            if len(auth) > 12:
                a_html = ", ".join(html.escape(a) for a in auth[:12]) + ", et al."
                full = (f'<details><summary>Full author list ({len(auth)})</summary>' +
                        ", ".join(html.escape(a) for a in auth) + "</details>")
            else:
                a_html = ", ".join(html.escape(a) for a in auth)
            items.append(f'<li id="ref-{k}">{a_html}. <em>{html.escape(m["title"])}</em>. '
                         f'arXiv:<a href="https://arxiv.org/abs/{aid}">{aid}</a> ({m["published"]}; read as {m["version"]}). '
                         f'{html.escape(note)} <span class="vtag">[R]</span>{full}</li>')
        elif k in REFS["manual"]:
            r = REFS["manual"][k]
            items.append(f'<li id="ref-{k}">{html.escape(", ".join(r["authors"]))}. <em>{html.escape(r["title"])}</em>. '
                         f'<a href="{r["url"]}">{html.escape(r["url"])}</a>. <span class="vtag">[{r["tag"]}]</span></li>')
        else:
            raise KeyError(f"unknown reference key {k}")
    legend = ("<p><strong>Verification tags.</strong> <span class=\"vtag\">[R]</span> full text obtained from arXiv and read "
              "(the four core papers end to end; for the others, the passages behind each cited fact). "
              "<span class=\"vtag\">[C]</span> code, config or model card read at the commit or date given. "
              "<span class=\"vtag\">[I]</span> indexed copy only: none in this version. Author lists are as returned by "
              "the arXiv API; team papers list the team as on arXiv.</p>")
    return legend + '<ol class="refs">' + "\n".join(items) + "</ol>"


def build(md_name, out_name, title, extra_refs=()):
    src = open(os.path.join(ROOT, md_name)).read()
    order = cite_numbers(src) + [k for k in extra_refs]
    num = {k: i + 1 for i, k in enumerate(order)}
    # protect code
    codes = []
    def keep(m):
        codes.append(m.group(0)); return f"\x00CODE{len(codes) - 1}\x00"
    src = re.sub(r"```.*?```", keep, src, flags=re.S)
    src = re.sub(r"`[^`\n]+`", keep, src)
    src = math(src)
    # citations
    def cite(m):
        ks = re.findall(r"@([\w-]+)", "@" + m.group(1))
        return '<span class="cite">[' + ", ".join(f'<a href="#ref-{k}">{num[k]}</a>' for k in ks) + "]</span>"
    src = re.sub(r"\[@([^\]]+)\]", cite, src)
    # tags
    src = re.sub(r"\[(proven|paper|config|measured|inference)((?::|,| )[^\]\[]*)?\]",
                 lambda m: f'<span class="tag tag-{m.group(1)}">{m.group(1)}{html.escape(m.group(2) or "")}</span>', src)
    # figures
    def fig(m):
        cap = m.group(3)
        head, _, rest = cap.partition(" ")
        head2, _, rest2 = rest.partition(" ") if head == "Figure" else ("", "", rest)
        label = f"{head} {head2}" if head == "Figure" else ""
        body = rest2 if head == "Figure" else cap
        return (f'<figure><div class="plate"><img alt="{html.escape(m.group(1))}" src="{m.group(2)}" loading="lazy"></div>'
                f'<figcaption><strong>{html.escape(label)}</strong> {html.escape(body)}</figcaption></figure>')
    src = re.sub(r'^!\[([^\]]*)\]\(([^ )]+) "([^"]+)"\)\s*$', fig, src, flags=re.M)
    src = src.replace("{{REFERENCES}}", "\x00REFS\x00")
    for i, c in enumerate(codes):
        src = src.replace(f"\x00CODE{i}\x00", c)
    body = markdown.markdown(src, extensions=["tables", "fenced_code"])
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    body = body.replace("<p>\x00REFS\x00</p>", render_refs(order)).replace("\x00REFS\x00", render_refs(order))
    page = HEAD.format(title=title, css=CSS) + "<main>\n" + body + "\n</main>\n"
    with open(os.path.join(SITE, out_name), "w") as f:
        f.write(page)
    print("wrote", os.path.join(SITE, out_name), f"{len(page)/1e3:.0f} kB, {len(order)} references")


if __name__ == "__main__":
    build("report.md", "report.html", "Where Softmax Lives")
    build("blog.md", "blog.html", "Softmax Across Architectures")

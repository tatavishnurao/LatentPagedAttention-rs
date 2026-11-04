"""Render paper.md, report.md and blog.md to site/*.html.

* $...$ / $$...$$ -> MathML (latex2mathml); code spans and blocks are protected first.
* ![alt](src "Figure N. caption") on its own line -> <figure> with caption.
* Citations: [@key] or [@key, locator; @key2, locator2] -> numbered links [n] to the reference list. Each link
  carries the locator; hovering, focusing or tapping it opens a preview card built from embedded JSON (title,
  authors, venue and year, a summary written from the full text, the cited section/table/figure, links).
  On touch screens the first tap opens the card and the card holds the jump link; on mouse and keyboard the
  link itself jumps to the reference list.
* {{REFERENCES}} -> reference list from sources/references.json + sources/arxiv_meta.json (full author lists).
* [proven] [paper] [config] [measured] [inference] -> tag chips (used in the technical report only).
Requires: markdown, latex2mathml."""
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
/* Layout: one reading column (~72ch); figures may run wider; technical-paper styling with citation preview cards. */
:root{
  --bg:#fbfbf9; --fg:#17191e; --muted:#596070; --rule:#dcdfe4; --accent:#1f5ea8;
  --panel:#eff2f6; --plate:#ffffff; --code-bg:#edf0f4; --card:#ffffff; --card-shadow:rgba(20,30,50,.16);
  --t-proven:#1d6b46; --t-paper:#1f5ea8; --t-config:#6a4c93; --t-measured:#9a5b00; --t-inference:#8a3a3a;
  --f-body:"Source Serif 4", Georgia, "Times New Roman", serif;
  --f-head:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif;
  --f-mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#14161a; --fg:#e6e7ea; --muted:#a2a8b3; --rule:#2d3139; --accent:#86b6f0;
  --panel:#1c2026; --plate:#ffffff; --code-bg:#1f232a; --card:#1d2128; --card-shadow:rgba(0,0,0,.5);
  --t-proven:#6fd0a0; --t-paper:#86b6f0; --t-config:#c3a6ee; --t-measured:#f0b866; --t-inference:#f09a9a; color-scheme:dark}}
:root[data-theme="dark"]{
  --bg:#14161a; --fg:#e6e7ea; --muted:#a2a8b3; --rule:#2d3139; --accent:#86b6f0;
  --panel:#1c2026; --plate:#ffffff; --code-bg:#1f232a; --card:#1d2128; --card-shadow:rgba(0,0,0,.5);
  --t-proven:#6fd0a0; --t-paper:#86b6f0; --t-config:#c3a6ee; --t-measured:#f0b866; --t-inference:#f09a9a; color-scheme:dark}
body{background:var(--bg);color:var(--fg);font:17px/1.62 var(--f-body);
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:40px 80px}
main{max-width:72ch;margin:0 auto;min-width:0}
h1,h2,h3{font-family:var(--f-head);line-height:1.2;text-wrap:balance;letter-spacing:.005em}
h1{font-size:2.05rem;margin:0 0 .3rem;font-weight:600}
h2{font-size:1.4rem;margin:2.4rem 0 .6rem;padding-top:1rem;border-top:1px solid var(--rule);font-weight:600}
h3{font-size:1.1rem;margin:1.7rem 0 .4rem;font-weight:600}
h1+p strong{font-family:var(--f-head);font-weight:500;color:var(--muted);font-size:1.12rem}
a{color:var(--accent);text-underline-offset:2px;overflow-wrap:anywhere}
a:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:2px}
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
a.cref{text-decoration:none;padding:0 .08em}
a.cref:hover{text-decoration:underline}
ol.refs{font-size:.88rem;line-height:1.45;padding-left:2.2em}
ol.refs li{margin:.5rem 0;scroll-margin-top:24px}
ol.refs li:target{background:var(--panel);border-radius:4px}
ol.refs details{margin-top:.2rem;font-size:.82rem;color:var(--muted)}
ol.refs summary{cursor:pointer;color:var(--accent)}
.vtag{font-family:var(--f-mono);font-size:.78em;color:var(--muted)}
.backlink{font-size:.9rem;color:var(--muted)}
.cite-card{position:absolute;z-index:20;box-sizing:border-box;width:min(380px,calc(100vw - 32px));background:var(--card);color:var(--fg);
  border:1px solid var(--rule);border-radius:8px;box-shadow:0 8px 28px var(--card-shadow);padding:12px 14px;
  font:14px/1.45 var(--f-body)}
.cite-card .cc-title{font-family:var(--f-head);font-weight:600;font-size:15px;line-height:1.25;margin:0 0 4px}
.cite-card .cc-meta{color:var(--muted);font-size:12.5px;margin:0 0 8px}
.cite-card .cc-sum{margin:0 0 8px}
.cite-card .cc-loc{margin:0 0 8px;font-size:13px}
.cite-card .cc-loc span{font-family:var(--f-mono);font-size:11.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.cite-card .cc-links{display:flex;gap:14px;flex-wrap:wrap;font-family:var(--f-head);font-size:13px}
@media (max-width:520px){body{font-size:16px} h1{font-size:1.7rem}}
@media (prefers-reduced-motion: reduce){*{scroll-behavior:auto}}
"""

JS = r"""
(function(){
  var data = JSON.parse(document.getElementById('refdata').textContent);
  var card = document.createElement('div');
  card.className = 'cite-card'; card.id = 'cite-card'; card.setAttribute('role','tooltip'); card.hidden = true;
  document.body.appendChild(card);
  var current = null, hideTimer = null;
  var touch = window.matchMedia('(hover: none)').matches;
  function esc(s){ return String(s).replace(/[&<>"]/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }
  function show(a){
    clearTimeout(hideTimer);
    var r = data[a.dataset.ref]; if(!r) return;
    var loc = a.dataset.loc ? '<p class="cc-loc"><span>Cited for</span><br>' + esc(a.dataset.loc) + '</p>' : '';
    card.innerHTML = '<p class="cc-title">' + esc(r.title) + '</p>' +
      '<p class="cc-meta">' + esc(r.authors) + ' · ' + esc(r.venue) + '</p>' +
      '<p class="cc-sum">' + esc(r.summary) + '</p>' + loc +
      '<div class="cc-links"><a href="#ref-' + esc(a.dataset.ref) + '">Go to reference [' + esc(a.textContent) + ']</a>' +
      (r.url ? '<a href="' + esc(r.url) + '" target="_blank" rel="noopener">Open source</a>' : '') + '</div>';
    card.hidden = false; current = a;
    a.setAttribute('aria-describedby', 'cite-card');
    var b = a.getBoundingClientRect(), w = card.offsetWidth, h = card.offsetHeight;
    var left = Math.min(Math.max(8, b.left + window.scrollX - 20), window.scrollX + document.documentElement.clientWidth - w - 8);
    var top = b.bottom + window.scrollY + 8;
    if (b.bottom + h + 16 > window.innerHeight && b.top > h + 16) top = b.top + window.scrollY - h - 8;
    card.style.left = left + 'px'; card.style.top = top + 'px';
  }
  function hide(){ card.hidden = true; if (current) current.removeAttribute('aria-describedby'); current = null; }
  function later(){ clearTimeout(hideTimer); hideTimer = setTimeout(function(){
    if (!card.matches(':hover') && !card.contains(document.activeElement) && document.activeElement !== current) hide(); }, 160); }
  document.querySelectorAll('a.cref').forEach(function(a){
    a.addEventListener('mouseenter', function(){ if(!touch) show(a); });
    a.addEventListener('mouseleave', later);
    a.addEventListener('focus', function(){ show(a); });
    a.addEventListener('blur', later);
    a.addEventListener('pointerdown', function(){ a._wasOpen = (current === a && !card.hidden); });
    a.addEventListener('click', function(e){ if (touch && !a._wasOpen){ e.preventDefault(); show(a); } });
  });
  card.addEventListener('mouseleave', later);
  card.addEventListener('focusout', later);
  card.addEventListener('click', function(e){ if (e.target.closest('a[href^="#ref-"]')) hide(); });
  document.addEventListener('keydown', function(e){ if (e.key === 'Escape' && current){ var c = current; hide(); c.focus(); } });
  document.addEventListener('click', function(e){ if (!card.contains(e.target) && !e.target.closest('a.cref')) hide(); });
  window.addEventListener('resize', hide);
})();
"""

HEAD = """<title>{title}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<style>{css}</style>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">
"""

CITE_RE = re.compile(r"\[@([^\]]+)\]")


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
    return re.sub(r"(?<![\\$])\$([^$\n]+?)\$", inline, src)


def parse_cite(body):
    """'@a, §2; @b' -> [('a', '§2'), ('b', '')]"""
    out = []
    body = "@" + body if not body.lstrip().startswith("@") else body
    for part in body.split(";"):
        part = part.strip()
        m = re.match(r"@([\w-]+)\s*,?\s*(.*)$", part)
        if not m:
            raise ValueError("bad citation: " + body)
        out.append((m.group(1), m.group(2).strip()))
    return out


def ref_record(k):
    r = REFS["refs"][k]
    if "arxiv" in r:
        m = META[r["arxiv"]]
        auth = [a.strip() for a in m["authors"] if a.strip() not in (":",)]
        short = ", ".join(auth[:3]) + (" et al." if len(auth) > 3 else "")
        return dict(title=m["title"], authors=short, full_authors=auth, venue=(r["venue"] if re.search(r"\b(19|20)\d\d\b", r["venue"]) else f'{r["venue"]}, arXiv {m["published"][:4]}'),
                    summary=r["summary"], url=f'https://arxiv.org/abs/{r["arxiv"]}', arxiv=r["arxiv"],
                    version=m["version"], published=m["published"], tag="R")
    return dict(title=r["title"], authors=", ".join(r["authors"]), full_authors=r["authors"], venue=r["venue"],
                summary=r["summary"], url=r["url"], tag=r.get("tag", "C"))


def render_refs(order):
    items = []
    for k in order:
        r = ref_record(k)
        auth = r["full_authors"]
        full = ""
        if len(auth) > 12:
            a_html = ", ".join(html.escape(a) for a in auth[:12]) + ", et al."
            full = (f'<details><summary>Full author list ({len(auth)})</summary>' +
                    ", ".join(html.escape(a) for a in auth) + "</details>")
        else:
            a_html = ", ".join(html.escape(a) for a in auth)
        if "arxiv" in r:
            where = (f'{html.escape(r["venue"])}. arXiv:<a href="{r["url"]}">{r["arxiv"]}</a> '
                     f'(first version {r["published"]}; read as {r["version"]}).')
        else:
            where = f'{html.escape(r["venue"])}. <a href="{r["url"]}">{html.escape(r["url"])}</a>.'
        items.append(f'<li id="ref-{k}">{a_html}. <em>{html.escape(r["title"])}</em>. {where} '
                     f'<span class="vtag">[{r["tag"]}]</span>{full}</li>')
    legend = ('<p><strong>Verification.</strong> <span class="vtag">[R]</span> full text read from arXiv in the '
              'session that wrote this page; <span class="vtag">[C]</span> code, config or model card read at the commit or '
              'date given. No reference rests on a search snippet. Author lists come from the arXiv API.</p>')
    return legend + '<ol class="refs">' + "\n".join(items) + "</ol>"


def build(md_name, out_name, title):
    src = open(os.path.join(ROOT, md_name)).read()
    # Python-Markdown needs a blank line before a list that follows a paragraph line
    src = re.sub(r"(?m)^(?!\s*(?:[-*]|\d+\.)\s)(?!\s*$)(?!\|)(?!>)(.+)\n(\s*(?:[-*]|\d+\.)\s)", r"\1\n\n\2", src)
    order = []
    for m in CITE_RE.finditer(src):
        for k, _ in parse_cite(m.group(1)):
            if k not in REFS["refs"]:
                raise KeyError(f"{md_name}: unknown reference key {k}")
            if k not in order:
                order.append(k)
    num = {k: i + 1 for i, k in enumerate(order)}
    codes = []

    def keep(m):
        codes.append(m.group(0))
        return f"\x00CODE{len(codes) - 1}\x00"
    src = re.sub(r"```.*?```", keep, src, flags=re.S)
    src = re.sub(r"`[^`\n]+`", keep, src)
    src = math(src)

    def cite(m):
        links = []
        for k, loc in parse_cite(m.group(1)):
            links.append(f'<a class="cref" href="#ref-{k}" data-ref="{k}" data-loc="{html.escape(loc)}">{num[k]}</a>')
        return '<span class="cite">[' + ", ".join(links) + "]</span>"
    src = CITE_RE.sub(cite, src)
    src = re.sub(r"\[(proven|paper|config|measured|inference)((?::|,| )[^\]\[]*)?\]",
                 lambda m: f'<span class="tag tag-{m.group(1)}">{m.group(1)}{html.escape(m.group(2) or "")}</span>', src)

    def fig(m):
        cap = m.group(3)
        mm = re.match(r"(Figure \d+\.)\s*(.*)$", cap, flags=re.S)
        label, body = (mm.group(1), mm.group(2)) if mm else ("", cap)
        return (f'<figure><div class="plate"><img alt="{html.escape(m.group(1))}" src="{m.group(2)}" loading="lazy"></div>'
                f'<figcaption><strong>{html.escape(label)}</strong> {html.escape(body)}</figcaption></figure>')
    src = re.sub(r'^!\[([^\]]*)\]\(([^ )]+) "([^"]+)"\)\s*$', fig, src, flags=re.M)
    src = src.replace("{{REFERENCES}}", "\x00REFS\x00")
    for i, c in enumerate(codes):
        src = src.replace(f"\x00CODE{i}\x00", c)
    body = markdown.markdown(src, extensions=["tables", "fenced_code"])
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    refs_html = render_refs(order)
    body = body.replace("<p>\x00REFS\x00</p>", refs_html).replace("\x00REFS\x00", refs_html)
    cards = {k: {f: v for f, v in ref_record(k).items() if f in ("title", "authors", "venue", "summary", "url")} for k in order}
    page = (HEAD.format(title=title, css=CSS) + "<main>\n" + body + "\n</main>\n" +
            '<script type="application/json" id="refdata">' + json.dumps(cards).replace("</", "<\\/") + "</script>\n" +
            "<script>" + JS + "</script>\n")
    with open(os.path.join(SITE, out_name), "w") as f:
        f.write(page)
    print("wrote", os.path.join(SITE, out_name), f"{len(page) / 1e3:.0f} kB, {len(order)} references")
    return order


if __name__ == "__main__":
    import sys
    pages = {"paper.md": ("paper.html", "Softmax in LLM Serving"),
             "report.md": ("report.html", "Where Softmax Lives v3"),
             "blog.md": ("blog.html", "Checking Softmax Claims")}
    for md, (out, title) in pages.items():
        if os.path.exists(os.path.join(ROOT, md)):
            build(md, out, title)

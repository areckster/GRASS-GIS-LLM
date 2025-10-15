#!/usr/bin/env python3
import os
import json
import re
import shlex
import hashlib
import argparse
from typing import List, Tuple, Optional, Dict
from html import unescape as html_unescape


# Try to use BeautifulSoup if available. Fallback to stdlib HTMLParser otherwise.
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:  # pragma: no cover
    BeautifulSoup = None  # type: ignore

# System prompt for fine-tuning dataset
SYSTEM_PROMPT = (
    "You are Grassy, a GRASS GIS expert developed by Areck.\n"
    "Write correct, executable Python using grass.script (and grass.tools when appropriate).\n"
    "Prefer gs.run_command / gs.read_command / gs.write_command.\n"
    "Only include runnable code inside <execute_grass> tags and keep it self-contained.\n"
    "Always:\n"
    "- import grass.script as gs\n"
    "- set region when needed (g.region ...)\n"
    "- start a display monitor before any d.* command: gs.run_command(\"d.mon\", start=\"cairo\") and stop it after with gs.run_command(\"d.mon\", stop=\"cairo\")\n"
    "- pass all required parameters for a module (e.g., d.correlate needs two rasters)\n"
    "- use realistic map names and paths (e.g., elevation, landuse96_28m, /data/dem.tif)\n"
    "- use overwrite=True for repeatable outputs when applicable\n\n"
    "User has the following data in their project:\n"
    "- rasters: \"dem\", \"elevation\", \"landuse96_28m\", \"slope\"\n"
    "- vectors: \"streets_wake\", \"nc_state\", \"urbanarea\", \"railroads\", \"hospitals\""
)

# Global counter to balance API exposure across samples
EXAMPLE_COUNTER = 0


def _md5_index(key: str, n: int) -> int:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(1, n)


CANONICAL_PREFIXES = ("d", "r", "v", "g", "i", "db", "ps", "m", "r3", "t")


def is_canonical_module(name: str) -> bool:
    if "." not in name:
        return False
    if not re.match(r"^[a-z0-9_.]+$", name):
        return False
    prefix = name.split(".", 1)[0]
    return prefix in CANONICAL_PREFIXES


USER_TEMPLATES = [
    "Use {module} as shown in the documentation.",
    "Reproduce the manual example for {module}.",
    "Demonstrate {module}: {context}",
    "Show how to {desc} using {module}.",
    "Set up and run {module} to {desc}.",
    "Using GRASS Python, perform: {context}",
    "Please run {module} to {desc}.",
    "How do I {desc}? Use {module}.",
    "With GRASS Python, execute {module} — goal: {desc}.",
    "Help: I need to {desc} via {module}.",
    "Use the display module {module} to {desc}.",
    "Use the raster module {module} to {desc}.",
]

FINAL_TEMPLATES = [
    "Task complete.",
    "Operation finished without errors.",
    "Module executed successfully.",
    "Successfully performed the requested operation.",
    "Completed without issues.",
    "Execution finished successfully.",
    "All steps ran as expected.",
    "Workflow completed without errors.",
    "Run finished cleanly.",
    "Processing concluded successfully.",
]


def _normalize_leading_verb_to_infinitive(desc: str) -> str:
    """Normalize the first word to base form for templates like
    'How do I {desc}?' or 'to {desc}'. Heuristic only.
    """
    if not desc:
        return desc
    desc = desc.strip()
    # Drop trailing period for insertion into sentences
    desc = desc[:-1] if desc.endswith('.') else desc
    parts = desc.split(maxsplit=1)
    first = parts[0]
    rest = parts[1] if len(parts) > 1 else ""
    lw = first.lower()
    # Skip if starts with 'to ' already
    if lw in {"to", "how", "use"}:
        return desc
    def de_s(word: str) -> str:
        wl = word.lower()
        if wl.endswith("ies") and len(wl) > 3:
            return wl[:-3] + "y"
        if wl.endswith("ses") or wl.endswith("xes") or wl.endswith("zes") or wl.endswith("ches") or wl.endswith("shes"):
            return wl[:-2]
        if wl.endswith("es") and (wl[-3] in "sxz" or wl.endswith("oes")):
            return wl[:-2]
        if wl.endswith("s") and not wl.endswith("ss"):
            return wl[:-1]
        return wl

    if "/" in lw:
        parts_slash = [de_s(x) for x in lw.split("/")]
        base = "/".join(parts_slash)
    else:
        base = de_s(lw)
    # Capitalization: keep lower-case start for mid-sentence phrases
    new_first = base
    normalized = (new_first + (" " + rest if rest else "")).strip()
    return normalized


def _realism_pass(py_code: str, module: str) -> str:
    """Replace placeholder values with realistic examples to improve signal."""
    # Generic replacements
    replacements = {
        "\"name\"": "\"elevation\"",
        "'name'": "'elevation'",
        "\"string\"": "\"elevation\"",
        "'string'": "'elevation'",
        "\"value\"": "\"500\"",
        "'value'": "'500'",
    }
    for k, v in replacements.items():
        py_code = py_code.replace(k, v)

    # Parameter-specific adjustments via simple regex
    # color=name -> color=red
    py_code = re.sub(r"color=\"name\"", "color=\"red\"", py_code)
    py_code = re.sub(r"color='name'", "color='red'", py_code)

    # frame=name -> frame=full_screen
    py_code = re.sub(r"frame=\"name\"", "frame=\"full_screen\"", py_code)
    py_code = re.sub(r"frame='name'", "frame='full_screen'", py_code)

    # raster=string -> elevation/landuse
    py_code = re.sub(r"raster=\"string\"", "raster=\"elevation\"", py_code)
    py_code = re.sub(r"raster='string'", "raster='elevation'", py_code)

    # map=name choose by family
    if module.startswith('r.') or module == 'd.rast':
        py_code = re.sub(r"map=\"name\"", "map=\"elevation\"", py_code)
        py_code = re.sub(r"map='name'", "map='elevation'", py_code)
    else:
        # Likely vector
        py_code = re.sub(r"map=\"name\"", "map=\"streets_wake\"", py_code)
        py_code = re.sub(r"map='name'", "map='streets_wake'", py_code)

    # input placeholders
    if module.startswith('r.'):
        py_code = re.sub(r"input=\"[^\"]*(path|file)[^\"]*\"", "input=\"/data/elevation.tif\"", py_code)
        py_code = re.sub(r"input='[^']*(path|file)[^']*'", "input='/data/elevation.tif'", py_code)
    elif module.startswith('v.'):
        py_code = re.sub(r"input=\"[^\"]*(path|file)[^\"]*\"", "input=\"/data/roads.shp\"", py_code)
        py_code = re.sub(r"input='[^']*(path|file)[^']*'", "input='/data/roads.shp'", py_code)

    return py_code


def _parse_with_bs4(html: str) -> Tuple[str, List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    # Prefer meta description
    meta_desc = soup.find("meta", attrs={"name": "description"})
    desc = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else ""
    # Fallback to DESCRIPTION section first paragraph
    if not desc:
        desc_h2 = soup.find(id="description") or soup.find("h2", string=lambda s: s and s.strip().upper()=="DESCRIPTION")
        if desc_h2:
            p = desc_h2.find_next("p")
            if p:
                desc = p.get_text(" ", strip=True)
    # Collect text of all <pre> blocks (raw, we will filter later)
    pres = [pre.get_text() for pre in soup.find_all("pre")]
    return desc, pres


def _parse_with_htmlparser(html: str) -> Tuple[str, List[str]]:
    from html.parser import HTMLParser
    from html import unescape

    class PreExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_pre = False
            self.in_p = False
            self.got_desc = False
            self.current_pre_parts: List[str] = []
            self.pre_texts: List[str] = []
            self.desc_parts: List[str] = []

        def handle_starttag(self, tag, attrs):
            if tag.lower() == "pre":
                self.in_pre = True
                self.current_pre_parts = []
            elif tag.lower() == "p" and not self.got_desc:
                self.in_p = True

        def handle_endtag(self, tag):
            if tag.lower() == "pre":
                self.in_pre = False
                txt = unescape("".join(self.current_pre_parts))
                self.pre_texts.append(txt)
                self.current_pre_parts = []
            elif tag.lower() == "p" and self.in_p:
                self.in_p = False
                if not self.got_desc:
                    # finalize description on closing the first <p>
                    self.got_desc = True

        def handle_data(self, data):
            if self.in_pre:
                self.current_pre_parts.append(data)
            elif self.in_p and not self.got_desc:
                self.desc_parts.append(data)

    parser = PreExtractor()
    parser.feed(html)
    desc = " ".join(" ".join(parser.desc_parts).split())
    return desc, parser.pre_texts


def parse_html(html: str) -> Tuple[str, List[str]]:
    if BeautifulSoup is not None:
        try:
            return _parse_with_bs4(html)
        except Exception:
            # Fallback if HTML is malformed for bs4
            return _parse_with_htmlparser(html)
    return _parse_with_htmlparser(html)


GRASS_PATTERNS = [
    r"\bgs\.",
    # Common in GRASS Python examples
    r"\bgscript\.",
    r"grass\.script",
    r"\bpygrass\b",
]


def looks_like_grass_python(code: str) -> bool:
    for pat in GRASS_PATTERNS:
        if re.search(pat, code):
            return True
    return False


def extract_examples_and_context(soup, module: str) -> List[Tuple[str, str]]:
    """Return list of (cli_text, context_text) from EXAMPLES section for the given module.
    Only examples that reference the module name are returned.
    """
    results: List[Tuple[str, str]] = []
    ex_h2 = soup.find(id="examples") or soup.find("h2", string=lambda s: s and s.strip().upper()=="EXAMPLES")
    if not ex_h2:
        return results
    # Stop at next h2
    section_nodes = []
    n = ex_h2
    while n is not None:
        n = n.find_next_sibling()
        if n and n.name == "h2":
            break
        if n is not None:
            section_nodes.append(n)

    for node in section_nodes:
        # Find code blocks within node
        for pre in node.find_all("pre"):
            code_text = pre.get_text("\n", strip=True)
            if module not in code_text:
                continue
            # Try to get a nearby description: prefer preceding h3 or p
            context = ""
            # Prefer the nearest previous h3/p regardless of nesting
            h3 = pre.find_previous("h3")
            if h3:
                context = h3.get_text(" ", strip=True)
            if not context:
                p = pre.find_previous("p")
                if p:
                    context = p.get_text(" ", strip=True)
            # Filter poor contexts like 'or'
            if context.strip().lower() in {"or", "example", "examples"} or len(context.strip()) <= 3:
                # Try one more previous paragraph
                p2 = (p.find_previous("p") if p else pre.find_previous("p"))
                if p2:
                    txt = p2.get_text(" ", strip=True)
                    if txt and txt.strip().lower() not in {"or", "example", "examples"}:
                        context = txt
            results.append((code_text, context))
    return results


def extract_parameters_summary(soup) -> List[str]:
    """Extract a concise list of key parameters from the 'Python (grass.script)' tab
    under Parameters section, returned as human-readable lines.
    """
    out: List[str] = []
    params_h2 = soup.find(id="parameters") or soup.find("h2", string=lambda s: s and s.strip().lower()=="parameters")
    if not params_h2:
        return out
    tabbed = params_h2.find_next("div", class_="tabbed-content")
    if not tabbed:
        return out
    blocks = tabbed.find_all("div", class_="tabbed-block")
    if len(blocks) < 2:
        return out
    # Second block is 'Python (grass.script)'
    b = blocks[1]
    p = b.find("p")
    if not p:
        return out
    # Each parameter starts with <strong>name</strong>
    for strong in p.find_all("strong"):
        name = strong.get_text(strip=True)
        # skip pseudo entries like 'flags'
        if not name or name.startswith("--"):
            continue
        # Find the following text up to a '<br>' as description line
        desc_text = ""
        # Gather next siblings until next <strong> or end
        sib = strong.next_sibling
        # Collate a short piece after the strong tag within same paragraph
        collected: List[str] = []
        while sib and not getattr(sib, 'name', None) == 'strong':
            txt = getattr(sib, 'get_text', lambda *a, **k: str(sib))()
            collected.append(txt)
            # Stop early after two breaks
            if len(" ".join(collected)) > 160:
                break
            sib = sib.next_sibling
        desc_text = re.sub(r"\s+", " ", "".join(collected)).strip(" :\n")
        if desc_text:
            out.append(f"{name}: {desc_text}")
        if len(out) >= 8:
            break
    return out


def extract_flag_descriptions(soup) -> Dict[str, str]:
    """From the 'Command line' tab, map single-letter flags to their meaning."""
    out: Dict[str, str] = {}
    params_h2 = soup.find(id="parameters") or soup.find("h2", string=lambda s: s and s.strip().lower()=="parameters")
    if not params_h2:
        return out
    tabbed = params_h2.find_next("div", class_="tabbed-content")
    if not tabbed:
        return out
    blocks = tabbed.find_all("div", class_="tabbed-block")
    if not blocks:
        return out
    b = blocks[0]  # command line
    p = b.find("p")
    if not p:
        return out
    # Patterns: <strong>-g</strong><br> Description text
    strongs = p.find_all("strong")
    for s in strongs:
        label = s.get_text(strip=True)
        if not label.startswith("-") or len(label) != 2:
            continue
        flag = label[1]
        # Collect the brief description following this strong tag
        desc_parts: List[str] = []
        sib = s.next_sibling
        # Collect up to the next <strong>
        while sib and getattr(sib, 'name', None) != 'strong':
            txt = getattr(sib, 'get_text', lambda *a, **k: str(sib))()
            desc_parts.append(txt)
            sib = sib.next_sibling
        desc = re.sub(r"\s+", " ", "".join(desc_parts)).strip(" :\n")
        if desc:
            out[flag] = desc
    return out


def fallback_between(html: str, start_marker: str, end_marker: str) -> Optional[str]:
    si = html.find(start_marker)
    if si == -1:
        return None
    ei = html.find(end_marker, si + len(start_marker))
    if ei == -1:
        ei = len(html)
    return html[si + len(start_marker):ei]


def fallback_extract_examples_and_context(html: str, module: str) -> List[Tuple[str, str]]:
    """Regex-based fallback to extract (cli_text, context) from EXAMPLES section."""
    results: List[Tuple[str, str]] = []
    section = fallback_between(html, '<h2 id="examples">', '<h2')
    if not section:
        return results
    # Split by pre blocks
    # Find preceding <p> just before each <pre>
    pattern = re.compile(r"((?:<h3[^>]*>.*?</h3>)|(?:<p>.*?</p>))?\s*(?:<div[^>]*>)?<pre[^>]*>([\s\S]*?)</pre>", re.IGNORECASE)
    for m in pattern.finditer(section):
        ctx_html = m.group(1) or ""
        code_html = m.group(2)
        code_text = html_unescape(re.sub(r"<[^>]+>", "", code_html))
        if module not in code_text:
            continue
        context = html_unescape(re.sub(r"<[^>]+>", "", ctx_html))
        context = re.sub(r"\s+", " ", context).strip()
        results.append((code_text.strip(), context))
    return results


def fallback_extract_parameters_summary(html: str) -> List[str]:
    out: List[str] = []
    # Grab Parameters section
    section = fallback_between(html, '<h2 id="parameters">', '<h2')
    if not section:
        return out
    # Within, find all <div class="tabbed-block"> blocks; pick the second one
    blocks = re.split(r"<div class=\"tabbed-block\">", section)
    if len(blocks) < 3:
        return out
    # The second block content is blocks[2-1]=blocks[1]
    b = blocks[1]
    # Extract <strong>param</strong> followed by brief text until <br>
    for sm in re.finditer(r"<strong>([^<]+)</strong>([^<]*)<br>([\s\S]*?)(?=(<strong>|$))", b):
        name = sm.group(1).strip()
        if not name or name.startswith("--"):
            continue
        # Take first two lines from description
        desc = re.sub(r"<[^>]+>", " ", sm.group(0))
        desc = html_unescape(desc)
        desc = re.sub(r"\s+", " ", desc)
        desc = desc.replace(name, "").strip(" :\n")
        if desc:
            out.append(f"{name}: {desc}")
        if len(out) >= 8:
            break
    return out


def fallback_extract_flag_descriptions(html: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    section = fallback_between(html, '<h2 id="parameters">', '<h2')
    if not section:
        return out
    blocks = re.split(r"<div class=\"tabbed-block\">", section)
    if not blocks:
        return out
    b = blocks[0]
    for sm in re.finditer(r"<strong>-(.)</strong>\s*<br>([\s\S]*?)(?=<strong>|$)", b):
        flag = sm.group(1)
        desc = re.sub(r"<[^>]+>", " ", sm.group(2))
        desc = re.sub(r"\s+", " ", desc).strip(" :\n")
        if desc:
            out[flag] = desc
    return out


def cli_here_doc_to_stdin(block_text: str, module: str) -> Optional[Tuple[str, str]]:
    """Parse a here-doc pattern: 'module [flags] << EOF ... EOF' and return
    (flags, stdin_string). Return None if not a match for this module.
    """
    # Normalize line endings
    text = block_text.strip("\n")
    # Simple regex capturing flags immediately after module
    m = re.search(
        rf"^\s*{re.escape(module)}\s+(-[A-Za-z]+)?\s*<<\s*EOF\s*\n([\s\S]*?)\nEOF\s*$",
        text,
        flags=re.M,
    )
    if not m:
        return None
    flags = m.group(1) or ""
    flags = flags.lstrip('-')
    stdin = m.group(2)
    return flags, stdin


def cli_line_to_python(module: str, line: str) -> Optional[Tuple[str, Dict[str, str], str, Optional[str]]]:
    """Convert one CLI line to (flags, kwargs, longflags) suitable for gs.run_command.
    longflags is a string like 'quiet'/'verbose' joined used for boolean keywords.
    Return None if the line cannot be parsed reliably.
    """
    try:
        tokens = shlex.split(line, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None
    # Handle echo ... | module ...
    if tokens[0] == 'echo' and '|' in tokens:
        try:
            pipe_index = tokens.index('|')
        except ValueError:
            pipe_index = -1
        if pipe_index > 0 and pipe_index < len(tokens) - 1 and tokens[pipe_index + 1] == module:
            stdin = tokens[1]
            # Collect short flags after module
            flags = ''
            longflags = []
            for t in tokens[pipe_index + 2:]:
                if t.startswith('--'):
                    longflags.append(t.lstrip('-'))
                elif t.startswith('-') and len(t) > 1:
                    flags += t.lstrip('-')
            return flags, {}, '|'.join(longflags), stdin
        return None

    if tokens[0] != module:
        return None
    flags = ''
    kwargs: Dict[str, str] = {}
    longflags: List[str] = []
    stdin_value: Optional[str] = None
    for t in tokens[1:]:
        if t.startswith('--'):
            lf = t.lstrip('-')
            # map --quiet -> quiet, --verbose -> verbose, ignore --help
            if lf in ('quiet', 'verbose', 'qq'):
                longflags.append(lf)
            continue
        if t.startswith('-') and len(t) > 1 and '=' not in t:
            # short flags can be combined, e.g., -gw
            flags += t.lstrip('-')
            continue
        if '=' in t:
            k, v = t.split('=', 1)
            kwargs[k] = v
            continue
        # Positional args are ambiguous, skip example if present
        return None
    return flags, kwargs, '|'.join(longflags), stdin_value


def build_python_from_cli(module: str, cli_block: str) -> Optional[str]:
    """Given a CLI code block that mentions the module, return a Python code string.
    Supports single-line invocations, echo piped stdin, and here-doc patterns.
    Returns None if conversion not possible.
    """
    # Here-doc pattern
    hd = cli_here_doc_to_stdin(cli_block, module)
    if hd is not None:
        flags, stdin = hd
        parts = ["import grass.script as gs", f"gs.write_command('{module}', flags='{flags}', stdin='''{stdin}''')"]
        return "\n".join(parts)

    # Multi-line: pick lines starting with module or echo ... | module
    lines = [ln.strip() for ln in cli_block.strip().splitlines() if ln.strip()]
    py_lines: List[str] = ["import grass.script as gs"]
    any_converted = False
    for ln in lines:
        conv = cli_line_to_python(module, ln)
        target = module
        if conv is None:
            # Try to parse a different GRASS module on this line (e.g., d.frame)
            try:
                tokens = shlex.split(ln, posix=True)
            except ValueError:
                tokens = []
            if tokens and re.match(r"^[a-z]\.[a-z0-9_.]+$", tokens[0]):
                target = tokens[0]
                conv = cli_line_to_python(target, ln)
        if conv is None:
            continue
        flags, kwargs, longflags, stdin_value = conv
        base_func = 'write_command' if stdin_value else 'run_command'
        kwlist = [f"flags='{flags}'"] if flags else []
        if longflags:
            for lf in longflags.split('|'):
                if not lf:
                    continue
                if lf == 'qq':
                    kwlist.append("superquiet=True")
                else:
                    kwlist.append(f"{lf}=True")
        for k, v in kwargs.items():
            v_escaped = v.replace("'", "\\'")
            kwlist.append(f"{k}='{v_escaped}'")
        if stdin_value:
            stdin_escaped = stdin_value.replace("'''", "'" + '"' * 3)
            kwlist.append(f"stdin='''{stdin_escaped}'''")
        py_lines.append(f"gs.{base_func}('{target}', {', '.join(kwlist)})")
        any_converted = True
    if any_converted:
        return "\n".join(py_lines)
    return None


def trim(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def parse_cli_for_summary(module: str, cli_block: str) -> Optional[Tuple[str, Dict[str, str], str, Optional[str]]]:
    for ln in [ln.strip() for ln in cli_block.strip().splitlines() if ln.strip()]:
        conv = cli_line_to_python(module, ln)
        if conv is not None:
            return conv
    return None


def build_user_from_cli(module: str, cli_block: str, fallback_user: str) -> str:
    conv = parse_cli_for_summary(module, cli_block)
    if conv is None:
        return fallback_user
    flags, kwargs, longflags, stdin_value = conv
    opts = ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else "none"
    outs = []
    for k in ("output", "out", "map", "raster", "vector", "slope", "aspect", "elevation"):
        if k in kwargs:
            outs.append(f"{k}={kwargs[k]}")
    outputs = ", ".join(outs) if outs else "unspecified"
    lines = [
        f"Task: Execute {module} to reproduce the documented example.",
        f"Flags: '{flags}'" if flags else "Flags: none",
        f"Options: {opts}",
        f"Outputs: {outputs}",
        "Use Python (grass.script).",
    ]
    if stdin_value:
        lines.append("Provide commands via stdin as shown.")
    return "\n".join(lines)


def vary_user_message(module: str, context: str, summary: str, cli_text: str, parsed: Optional[Tuple[str, Dict[str, str], str, Optional[str]]]) -> str:
    # Derive a short description from context or summary
    desc = context.strip() if context and len(context.strip()) > 3 else summary
    if desc:
        # remove module name and leading verbs where possible
        desc = re.sub(rf"\b{re.escape(module)}\b\s*", "", desc, flags=re.I).strip()
        # Lowercase first char for natural phrasing
        if desc and desc[0].isupper():
            desc = desc[0].lower() + desc[1:]
        # Normalize leading verb to base (fill, draw, set, generate, ...)
        desc = _normalize_leading_verb_to_infinitive(desc)
    else:
        desc = f"run {module} with appropriate options"

    # Pick template deterministically
    idx = _md5_index(module + "|" + cli_text, len(USER_TEMPLATES))
    template = USER_TEMPLATES[idx]
    text = template.format(module=module, context=context or summary or module, desc=desc)

    # If the chosen template still looks vague, fallback to CLI-derived summary
    if text.strip().lower() in {"or", "example", "examples"} or len(text.strip()) <= 3:
        text = build_user_from_cli(module, cli_text, text)
    # Minor wording nits
    text = re.sub(r"\beras\b", "erase", text)
    return text


def augment_python_code(module: str, py_code: str, kwargs: Dict[str, str], example_index: int) -> str:
    """Add helpful steps (g.region, r.colors, read_command) to make code richer."""
    lines = py_code.splitlines()
    if not lines:
        return py_code
    # Where to insert helper lines (after the import)
    insert_at = 1 if len(lines) > 1 and lines[0].startswith("import grass.script") else 0

    # Heuristic: for r.slope.aspect, ensure region and add r.colors for slope
    if module == 'r.slope.aspect':
        elev = kwargs.get('elevation')
        if elev and "g.region" not in py_code:
            lines.insert(insert_at, f"gs.run_command('g.region', raster='{elev}')")
            insert_at += 1
        slope_map = kwargs.get('slope')
        if slope_map:
            # Ensure color application after slope generated; append at end
            lines.append(f"gs.run_command('r.colors', map='{slope_map}', color='srtm')")

    # General raster heuristic: set region by raster if provided
    if module.startswith('r.'):
        for key in ('elevation', 'raster', 'input'):
            val = kwargs.get(key)
            if val and "g.region" not in py_code:
                lines.insert(insert_at, f"gs.run_command('g.region', raster='{val}')")
                insert_at += 1
                break

    # Balanced API exposure: add read_command every ~20 examples
    if 'read_command(' not in py_code and example_index % 20 == 7 and module not in ('d.fontlist',):
        # Add a g.region -p readout
        lines.insert(insert_at, "text = gs.read_command('g.region', flags='p')")
        insert_at += 1
        lines.insert(insert_at, "print(text)")
        insert_at += 1

    # Balanced API exposure: ensure some examples use write_command
    if 'write_command(' not in py_code and example_index % 20 == 13:
        # Draw a small plus icon at screen center using d.graph
        lines.append("gs.write_command('d.graph', stdin='''  color black\n  icon + 2 50 50''')")

    # Display monitor management for d.* modules (except pure d.colorlist or d.fontlist)
    d_call_pattern = re.compile(r"gs\.(?:run_command|write_command)\([\"'](d\.[^\"']+)[\"']")
    d_modules_used = []
    for ln in lines:
        m = d_call_pattern.search(ln)
        if m:
            d_modules_used.append(m.group(1))
    has_d_calls = bool(d_modules_used) or module.startswith('d.')
    has_d_mon = any("gs.run_command('d.mon'" in ln for ln in lines)
    only_d_colorlist = bool(d_modules_used) and set(d_modules_used) == {"d.colorlist"}
    only_d_fontlist = bool(d_modules_used) and set(d_modules_used) == {"d.fontlist"}

    # Transform d.colorlist to read_command (no monitor)
    if "d.colorlist" in d_modules_used or module == 'd.colorlist':
        new_lines = []
        replaced = False
        for ln in lines:
            if re.search(r"gs\.run_command\([\"']d\.colorlist[\"']", ln):
                new_lines.append("colors = gs.read_command('d.colorlist')")
                new_lines.append("print(colors)")
                replaced = True
            else:
                new_lines.append(ln)
        if replaced:
            lines = new_lines
            py_code = "\n".join(lines)

    # Transform d.fontlist to read_command (no monitor); remove unrelated region readouts
    if "d.fontlist" in d_modules_used or module == 'd.fontlist':
        new_lines = []
        replaced = False
        for ln in lines:
            if re.search(r"gs\.run_command\([\"']d\.fontlist[\"']", ln):
                new_lines.append("fonts = gs.read_command('d.fontlist')")
                new_lines.append("print(fonts)")
                replaced = True
            else:
                new_lines.append(ln)
        if replaced:
            lines = new_lines
            py_code = "\n".join(lines)
        # Remove unrelated g.region readout lines if present
        lines = [ln for ln in lines if "text = gs.read_command('g.region', flags='p')" not in ln and "print(text)" not in ln]

    # Insert region and monitor for display commands (skip pure d.colorlist-only scripts)
    has_other_d = has_d_calls and not (only_d_colorlist or only_d_fontlist)
    if has_other_d and not has_d_mon:
        # Set region to a sensible default if not specified
        if module != 'd.font' and "g.region" not in "\n".join(lines):
            lines.insert(insert_at, "gs.run_command('g.region', raster='elevation')")
            insert_at += 1
        # Start monitor
        mon_insert_at = insert_at
        # If a region line exists, start monitor after it
        for idx, ln in enumerate(lines):
            if "gs.run_command('g.region'" in ln:
                mon_insert_at = idx + 1
                break
        lines.insert(mon_insert_at, "gs.run_command('d.mon', start='cairo')")
        # Append stop at end
        lines.append("gs.run_command('d.mon', stop='cairo')")

    # Ensure d.legend.vect has a vector drawn first
    if any("gs.run_command('d.legend.vect'" in ln for ln in lines) and not any("gs.run_command('d.vect'" in ln for ln in lines):
        # Insert a simple d.vect before legend
        # If a monitor was just started, place after it
        insert_after = 0
        for idx, ln in enumerate(lines):
            if "gs.run_command('d.mon', start='cairo')" in ln:
                insert_after = idx + 1
                break
        lines.insert(insert_after, "gs.run_command('d.vect', map='streets_wake')")

    # Special handling for d.correlate: ensure two rasters and prepare slope
    d_corr_call = re.compile(r"gs\.run_command\((['\"])d\.correlate\1")
    if any(d_corr_call.search(ln) for ln in lines):
        # Ensure region
        if "g.region" not in "\n".join(lines):
            lines.insert(insert_at, "gs.run_command('g.region', raster='elevation')")
            insert_at += 1
        # Ensure slope exists
        slope_line = "gs.run_command('r.slope.aspect', elevation='elevation', slope='slope', overwrite=True)"
        if not any("gs.run_command('r.slope.aspect'" in ln for ln in lines):
            lines.insert(insert_at, slope_line)
            insert_at += 1
        # Ensure d.correlate has map='elevation,slope'
        new_lines = []
        for ln in lines:
            if d_corr_call.search(ln):
                if "map=" not in ln:
                    ln = "gs.run_command('d.correlate', map='elevation,slope')"
                else:
                    # Force two rasters in map parameter regardless of current value
                    ln = re.sub(r"map=([\"']).*?\1", "map='elevation,slope'", ln)
            new_lines.append(ln)
        lines = new_lines
        # Reorder so slope is computed before starting monitor
        try:
            mon_idx = next(i for i, ln in enumerate(lines) if re.search(r"gs\.run_command\((['\"])d\.mon\1, start=([\"'])cairo\2\)", ln))
        except StopIteration:
            mon_idx = None
        slope_idx = None
        for i, ln in enumerate(lines):
            if re.search(r"gs\.run_command\((['\"])r\.slope\.aspect\1", ln):
                slope_idx = i
                break
        if mon_idx is not None and slope_idx is not None and slope_idx > mon_idx:
            # move slope line to just before monitor start
            sl = lines.pop(slope_idx)
            lines.insert(mon_idx, sl)

    # d.font: ensure a concrete font and avoid forcing region
    d_font_call = re.compile(r"gs\.run_command\((['\"])d\.font\1")
    if any(d_font_call.search(ln) for ln in lines):
        # Ensure font parameter / replace bare call
        new_lines = []
        for ln in lines:
            if d_font_call.search(ln):
                new_lines.append("gs.run_command('d.font', font='DejaVuSans')")
            else:
                new_lines.append(ln)
        lines = new_lines

    # d.barscale: add placement and length for determinism
    # d.barscale: add placement and length for determinism (match single or double quotes)
    barscale_call = re.compile(r"gs\.run_command\((['\"])d\.barscale\1")
    if any(barscale_call.search(ln) for ln in lines):
        new_lines = []
        for ln in lines:
            if barscale_call.search(ln) and "at=" not in ln and "length=" not in ln:
                new_lines.append("gs.run_command('d.barscale', at='5,95', length='50', units='meters')")
            else:
                new_lines.append(ln)
        lines = new_lines

    # d.background: if starting monitor, use cairo output to a file for determinism
    if any(re.search(r"gs\.run_command\((['\"])d\.background\1", ln) for ln in lines):
        mon_start = re.compile(r"gs\.run_command\((['\"])d\.mon\1, start=([\"'])cairo\2\)")
        for i, ln in enumerate(lines):
            if mon_start.search(ln) and "output=" not in ln:
                lines[i] = "gs.run_command('d.mon', start='cairo', output='map.png', width=1024, height=768)"
                break

    # Add overwrite=True when an output is being produced
    new_lines = []
    for ln in lines:
        if ("gs.run_command('r." in ln or "gs.run_command('v." in ln or "gs.run_command('i." in ln or "gs.run_command('r3." in ln) and "output=" in ln and "overwrite=" not in ln:
            ln = ln.rstrip(')') + ", overwrite=True)"
        if "gs.run_command('r.slope.aspect'" in ln and "overwrite=" not in ln:
            ln = ln.rstrip(')') + ", overwrite=True)"
        new_lines.append(ln)
    lines = new_lines

    return "\n".join(lines)


def make_training_pairs(html: str, tool_name: str):
    global EXAMPLE_COUNTER
    # Summary/description
    summary = ""
    examples: List[Tuple[str, str]] = []
    key_params: List[str] = []
    flag_desc: Dict[str, str] = {}

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        meta_desc = soup.find("meta", attrs={"name": "description"})
        summary = meta_desc["content"].strip() if meta_desc and meta_desc.get("content") else ""
        if not summary:
            summary, _ = parse_html(html)
        examples = extract_examples_and_context(soup, tool_name)
        key_params = extract_parameters_summary(soup)
        flag_desc = extract_flag_descriptions(soup)
    else:
        # Fallback parsing without BeautifulSoup
        # Meta description
        m = re.search(r"<meta[^>]+name=\"description\"[^>]+content=\"(.*?)\"", html, re.IGNORECASE)
        if m:
            summary = html_unescape(m.group(1))
        if not summary:
            summary, _ = parse_html(html)
        examples = fallback_extract_examples_and_context(html, tool_name)
        key_params = fallback_extract_parameters_summary(html)
        flag_desc = fallback_extract_flag_descriptions(html)
    if not examples:
        # Fallback to any Python snippet already present
        _, pre_blocks = parse_html(html)
        for code in pre_blocks:
            if looks_like_grass_python(code):
                # Ensure uniform import
                if 'import grass.script as gs' not in code:
                    code = 'import grass.script as gs\n' + code
                # Augment and realism passes
                code = augment_python_code(tool_name, code, {}, EXAMPLE_COUNTER)
                code = _realism_pass(code, tool_name)
                # Vary user phrasing even in fallback
                user_task = vary_user_message(tool_name, "", summary, tool_name + "|py", None)
                user = f"{user_task}\nModule: {tool_name}"
                # Module-specific override messages for clarity
                if tool_name == 'd.correlate':
                    think_msg = "d.correlate plots correlation between two rasters. I’ll set the region to the DEM, compute slope from elevation, then correlate elevation vs slope."
                    final_msg = "Correlation plotted for elevation vs slope."
                elif tool_name == 'd.font':
                    think_msg = "Select a concrete font for subsequent display text; display commands require an active monitor."
                    final_msg = "Display font set to DejaVuSans."
                elif tool_name == 'd.fontlist':
                    think_msg = "List available display fonts by capturing stdout from d.fontlist."
                    final_msg = "Available fonts listed to stdout."
                else:
                    think_msg = f"From GRASS docs: {tool_name} — {trim(summary, 220)}"
                    final_msg = FINAL_TEMPLATES[_md5_index(tool_name + '|fallback', len(FINAL_TEMPLATES))]

                assistant = (
                    f"<think>{think_msg}</think>"
                    f"<execute_grass>{code}</execute_grass>"
                    f"<final>{final_msg}</final>"
                )
                yield {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ]
                }
                EXAMPLE_COUNTER += 1
        return

    for cli_text, context in examples:
        py_code = build_python_from_cli(tool_name, cli_text)
        if not py_code:
            continue
        # Compose a more specific user prompt
        user_task = vary_user_message(tool_name, context, summary, cli_text, None)
        user = f"{user_task}\nModule: {tool_name}"

        # Extract flags actually used to describe them briefly
        m = re.findall(rf"{re.escape(tool_name)}\s+(-[A-Za-z]+)", cli_text)
        used_flags = ''.join([x.lstrip('-') for x in m]) if m else ''
        flag_lines = []
        for ch in used_flags:
            if ch in flag_desc:
                flag_lines.append(f"-{ch}: {flag_desc[ch]}")
        flag_info = (" Flags: " + "; ".join(flag_lines)) if flag_lines else ""

        # Build THINK section with summary, plan, and key params
        think_lines: List[str] = []
        if summary:
            think_lines.append(f"{tool_name}: {trim(summary, 220)}")
        # Heuristic hints by module family
        family = tool_name.split('.', 1)[0] if '.' in tool_name else ''
        if family == 'd':
            think_lines.append("Note: Requires active display monitor (d.mon).")
        elif family == 'r':
            think_lines.append("Raster module: ensure region/resolution are set (g.region).")
        elif family == 'v':
            think_lines.append("Vector module: consider topology checks (v.build).")
        elif family == 'i':
            think_lines.append("Imagery module: use groups/targets where applicable.")
        if key_params:
            think_lines.append("Key parameters: " + "; ".join([trim(x, 100) for x in key_params[:10]]))
        if flag_info:
            think_lines.append(flag_info)
        # Add procedural plan based on parsed CLI
        summary_conv = parse_cli_for_summary(tool_name, cli_text)
        if summary_conv is not None:
            _, kwargs_sum_plan, _, stdin_plan = summary_conv
            plan_bits = []
            fam = tool_name.split('.', 1)[0] if '.' in tool_name else ''
            if fam == 'd':
                plan_bits.append("start a cairo monitor")
            if fam == 'r' and ('elevation' in kwargs_sum_plan or 'raster' in kwargs_sum_plan):
                plan_bits.append("set the region to the input raster")
            if stdin_plan:
                plan_bits.append("feed commands via stdin")
            plan_bits.append(f"run {tool_name} with the provided options")
            think_lines.append("Plan: " + ", ".join(plan_bits) + ".")
        # Include compact CLI preview
        cli_preview = trim(re.sub(r"\s+", " ", cli_text.replace('\n', ' ')), 240)
        think_lines.append(f"CLI: {cli_preview}")
        think = trim(" ".join(think_lines), 800)

        # Augment code for depth and balance
        if summary_conv is not None:
            _, kwargs_sum, _, _ = summary_conv
        else:
            kwargs_sum = {}
        py_code = augment_python_code(tool_name, py_code, kwargs_sum, EXAMPLE_COUNTER)
        py_code = _realism_pass(py_code, tool_name)

        # Compose final message, including outputs if present and varied phrasing
        final_note = f"Completed task using {tool_name}."
        outs: List[str] = []
        if summary_conv is not None:
            _, kwargs_sum_final, _, _ = summary_conv
            for k in ("output", "out", "map", "raster", "vector", "slope", "aspect"):
                if k in kwargs_sum_final:
                    outs.append(f"{k}={kwargs_sum_final[k]}")
        if outs:
            final_note = f"Completed {tool_name}; outputs: {', '.join(outs)}."
        else:
            idx_final = _md5_index(tool_name + "|" + cli_text, len(FINAL_TEMPLATES))
            final_note = FINAL_TEMPLATES[idx_final]

        # Module-specific overrides
        if tool_name == 'd.correlate':
            think = "d.correlate plots correlation between two rasters. I’ll set the region to the DEM, compute slope from elevation, then correlate elevation vs slope."
            final_note = "Correlation plotted for elevation vs slope."
        elif tool_name == 'd.font':
            think = "Select a concrete font for subsequent display text; display commands require an active monitor."
            final_note = "Display font set to DejaVuSans."
        elif tool_name == 'd.fontlist':
            think = "List available display fonts by capturing stdout from d.fontlist."
            final_note = "Available fonts listed to stdout."

        assistant = (
            f"<think>{think}</think>"
            f"<execute_grass>{py_code}</execute_grass>"
            f"<final>{final_note}</final>"
        )
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
        EXAMPLE_COUNTER += 1


def read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            return f.read()


def generate_dataset(folder: str) -> list:
    records = []
    for name in sorted(os.listdir(folder)):
        if not name.lower().endswith(".html"):
            continue
        fpath = os.path.join(folder, name)
        if not os.path.isfile(fpath):
            continue
        html = read_text(fpath)
        tool = os.path.splitext(name)[0]
        # Only include canonical GRASS modules (e.g., r.slope.aspect, v.overlay)
        if not is_canonical_module(tool):
            continue
        records.extend(make_training_pairs(html, tool))
    return records


def main():
    ap = argparse.ArgumentParser(description="Generate GRASS GIS fine-tune dataset from HTML docs")
    ap.add_argument("--folder", default=".", help="Folder containing .html files (default: .)")
    ap.add_argument(
        "--out",
        default="grass_finetune_dataset.jsonl",
        help="Output JSONL path (default: grass_finetune_dataset.jsonl)",
    )
    args = ap.parse_args()

    records = generate_dataset(args.folder)
    with open(args.out, "w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()

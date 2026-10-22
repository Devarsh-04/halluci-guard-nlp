"""Build data/visa_home_affairs.jsonl from official Department of Home Affairs
visa pages. Run from the repository root: python3 data/build_visa_kb.py

Content: Australian Government Department of Home Affairs, licensed under
Creative Commons Attribution 3.0 Australia (CC BY 3.0 AU).
"""
import datetime
import html
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/"
PAGES = {
    "student-500": "Student visa (subclass 500)",
    "visitor-600/tourist-stream-overseas": "Visitor visa (subclass 600), Tourist stream (apply outside Australia)",
    "visitor-600/tourist-stream-onshore": "Visitor visa (subclass 600), Tourist stream (apply in Australia)",
    "visitor-600/business-visitor-stream": "Visitor visa (subclass 600), Business Visitor stream",
    "visitor-600/sponsored-family-stream": "Visitor visa (subclass 600), Sponsored Family stream",
    "electronic-travel-authority-601": "Electronic Travel Authority (subclass 601)",
    "evisitor-651": "eVisitor (subclass 651)",
    "work-holiday-417/first-working-holiday-417": "Working Holiday visa (subclass 417), first Working Holiday visa",
    "work-holiday-417/second-working-holiday-417": "Working Holiday visa (subclass 417), second Working Holiday visa",
    "work-holiday-417/third-working-holiday-417": "Working Holiday visa (subclass 417), third Working Holiday visa",
    "temporary-graduate-485/post-higher-education-work": "Temporary Graduate visa (subclass 485), Post-Higher Education Work stream",
    "temporary-graduate-485/post-vocational-education-work": "Temporary Graduate visa (subclass 485), Post-Vocational Education Work stream",
    "skilled-independent-189/points-tested": "Skilled Independent visa (subclass 189), Points-tested stream",
    "skilled-nominated-190": "Skilled Nominated visa (subclass 190)",
    "skilled-work-regional-provisional-491/application": "Skilled Work Regional (Provisional) visa (subclass 491)",
    "partner-onshore/temporary-820": "Partner visa (temporary, subclass 820)",
    "partner-onshore/permanent-801": "Partner visa (permanent, subclass 801)",
    "partner-offshore/provisional-309": "Partner visa (provisional, subclass 309)",
    "partner-offshore/migrant-100": "Partner visa (migrant, subclass 100)",
    "work-holiday-462/first-work-holiday-462": "Work and Holiday visa (subclass 462), first Work and Holiday visa",
    "skills-in-demand-visa-subclass-482/core-skills-stream": "Skills in Demand visa (subclass 482), Core Skills stream",
    "skills-in-demand-visa-subclass-482/specialist-skills-stream": "Skills in Demand visa (subclass 482), Specialist Skills stream",
    "employer-nomination-scheme-186/temporary-residence-transition-stream": "Employer Nomination Scheme visa (subclass 186), Temporary Residence Transition stream",
    "employer-nomination-scheme-186/direct-entry-stream": "Employer Nomination Scheme visa (subclass 186), Direct Entry stream",
    "skilled-employer-sponsored-regional-494/employer-sponsored-stream": "Skilled Employer Sponsored Regional (Provisional) visa (subclass 494), Employer Sponsored stream",
    "prospective-marriage-300": "Prospective Marriage visa (subclass 300)",
    "parent-103": "Parent visa (subclass 103)",
    "contributory-parent-143": "Contributory Parent visa (subclass 143)",
    "aged-parent-804": "Aged Parent visa (subclass 804)",
    "sponsored-parent-temporary-870": "Sponsored Parent (Temporary) visa (subclass 870)",
    "transit-771": "Transit visa (subclass 771)",
    "medical-treatment-602": "Medical Treatment visa (subclass 602)",
}
# Phrases that identify each subclass in a question or answer.
ALIASES = {
    "500": ["student visa", "subclass 500"],
    "600": ["visitor visa", "tourist visa", "business visitor", "sponsored family", "subclass 600"],
    "601": ["electronic travel authority", "eta", "subclass 601"],
    "651": ["evisitor", "subclass 651"],
    "417": ["working holiday", "subclass 417"],
    "462": ["working holiday", "work and holiday", "subclass 462"],
    "485": ["temporary graduate", "post-higher education", "post-vocational", "graduate visa", "subclass 485"],
    "189": ["skilled independent", "subclass 189"],
    "190": ["skilled nominated", "subclass 190"],
    "491": ["skilled work regional", "subclass 491"],
    "820": ["partner visa", "subclass 820"],
    "801": ["partner visa", "subclass 801"],
    "309": ["partner visa", "subclass 309"],
    "100": ["partner visa", "subclass 100"],
    "482": ["skills in demand", "temporary skill shortage", "tss visa", "work visa", "subclass 482"],
    "186": ["employer nomination", "ens visa", "subclass 186"],
    "494": ["skilled employer sponsored regional", "subclass 494"],
    "300": ["prospective marriage", "fiance visa", "fiancé visa", "subclass 300"],
    "103": ["parent visa", "subclass 103"],
    "143": ["contributory parent", "parent visa", "subclass 143"],
    "804": ["aged parent", "parent visa", "subclass 804"],
    "870": ["sponsored parent", "parent visa", "subclass 870"],
    "771": ["transit visa", "subclass 771"],
    "602": ["medical treatment", "subclass 602"],
}
ATTRIBUTION = "Australian Government Department of Home Affairs, CC BY 3.0 AU"
SECTIONS = {
    "withThisVisa": "With this visa you can",
    "visaStay": "How long you can stay",
    "visaCost": "Cost",
    "notes": "You must",
}
TEXT_KEYS = ("block", "description", "subTitle", "body")
TITLE_KEYS = ("title", "heading", "name")
OUT = Path(__file__).with_name("visa_home_affairs.jsonl")


def fetch(url):
    request = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (HalluciGuard knowledge-base builder)",
    })
    return urllib.request.urlopen(request, timeout=60).read().decode("utf-8", "ignore")


def page_data(page):
    for match in re.finditer(r'="(\{&quot;[^"]*\})"', page):
        data = json.loads(html.unescape(match.group(1)))
        if "applicant" in data:
            return data
    return None


def sentences(fragment):
    # Prices are filled in by the browser, so their placeholder spans are empty here.
    fragment = re.sub(r"<span[^>]*data-svpattern[^>]*>.*?</span>", " [PRICE] ", fragment, flags=re.S)
    fragment = re.sub(r"</(li|p|h\d)>", ". ", fragment)
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment)).replace("​", "")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    text = re.sub(r"([.:;])\.", r"\1", text)
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", text):
        sentence = sentence.strip(" .") + "."
        if len(sentence) >= 8 and "[PRICE]" not in sentence and not re.match(r"(From )?\(", sentence):
            out.append(sentence)
    return out


def chunks(context, text, budget=600):
    """Group sentences up to a character budget and prefix each group with its
    visa and section, so a passage still says what it is about on its own."""
    groups, current = [], []
    for sentence in text:
        if current and len(" ".join(current + [sentence])) > budget:
            groups.append(current)
            current = []
        current.append(sentence)
    if current:
        groups.append(current)
    return [f"{context}: " + " ".join(group) for group in groups]


def sections(data):
    overview = data["applicant"].get("overview", {})
    for key, title in SECTIONS.items():
        if overview.get(key):
            yield title, overview[key]

    def walk(node):
        if isinstance(node, dict):
            title = next((node[k] for k in TITLE_KEYS if isinstance(node.get(k), str)), None)
            for key in TEXT_KEYS:
                if isinstance(node.get(key), str) and title:
                    yield re.sub(r"<[^>]+>", "", html.unescape(title)).strip(), node[key]
            for key, value in node.items():
                if key != "overview":
                    yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    yield from walk(data["applicant"])


def main():
    retrieved = datetime.date.today().isoformat()
    docs, seen = [], set()
    for slug, name in PAGES.items():
        url = BASE + slug
        try:
            page = fetch(url)
        except urllib.error.HTTPError as error:
            print(f"skip {slug}: HTTP {error.code}", file=sys.stderr)
            continue
        data = page_data(page)
        if data is None:
            print(f"skip {slug}: no visa content found", file=sys.stderr)
            continue
        subclass = re.search(r"\d{3}", slug).group(0)
        count = 0
        for number, (section, fragment) in enumerate(sections(data), 1):
            text = chunks(f"{name}, {section}", sentences(fragment))
            key = (name, " ".join(text))
            if not text or key in seen or section.startswith("New "):
                continue
            seen.add(key)
            for part, chunk in enumerate(text, 1):
                count += 1
                docs.append({
                    # A conflict only counts within one section: the id up to the part number.
                    "doc_id": f"ha-{slug.replace('/', '-')}-s{number:03d}-{part}",
                    "title": f"{name}: {section}",
                    "abstract": [chunk],
                    "subclass": subclass,
                    "aliases": ALIASES[subclass],
                    "publisher": "Department of Home Affairs",
                    "url": url,
                    "retrieved": retrieved,
                    "attribution": ATTRIBUTION,
                })
        print(f"{slug}: {count} passages", file=sys.stderr)
    OUT.write_text("".join(json.dumps(doc) + "\n" for doc in docs))
    print(f"wrote {len(docs)} passages to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
bklint - Bookmark Linter

Clean up and organize browser bookmarks from Firefox or Chrome.

Features:
- Auto-detect browser format (Firefox JSON/HTML, Chrome JSON)
- Remove duplicate bookmarks
- Check for dead/broken links
- Smart categorization using OpenDNS domain tagging
- Reorganize into cleaner category structure
- Export to multiple formats
"""

import asyncio
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from urllib.parse import urlparse

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.prompt import Confirm, Prompt

console = Console()

# Cache file for OpenDNS categories
CACHE_FILE = Path(".bklint-category-cache.json")


class BrowserFormat(Enum):
    FIREFOX_JSON = "firefox_json"
    FIREFOX_HTML = "firefox_html"
    CHROME_JSON = "chrome_json"
    UNKNOWN = "unknown"


# OpenDNS category mapping to our folder names
# Maps OpenDNS categories to simplified folder names
OPENDNS_CATEGORY_MAP = {
    # Technology & Development
    "Software/Technology": "Software & Technology",
    "Business Services": "Business Services",
    "SaaS and B2B": "Software & Technology",

    # Entertainment & Media
    "Video Sharing": "Video & Streaming",
    "Music": "Music",
    "Movies": "Movies & TV",
    "Television": "Movies & TV",
    "Radio": "Music",
    "Podcasts": "Podcasts",
    "Anime/Manga/Webcomic": "Entertainment",
    "Humor": "Entertainment",
    "Games": "Gaming",

    # Social & Communication
    "Social Networking": "Social Media",
    "Chat": "Communication",
    "Instant Messaging": "Communication",
    "Webmail": "Communication",
    "Forums/Message boards": "Forums & Communities",
    "Blogs": "Blogs",

    # Shopping & Business
    "Ecommerce/Shopping": "Shopping",
    "Auctions": "Shopping",
    "Classifieds": "Shopping",
    "Financial Institutions": "Finance",
    "Automotive": "Automotive",
    "Real Estate": "Real Estate",

    # Information & Reference
    "News/Media": "News",
    "Educational Institutions": "Education",
    "Research/Reference": "Reference",
    "Search Engines": "Search Engines",
    "Portals": "Portals",
    "Government": "Government",
    "Religious": "Religion",
    "Non-Profits": "Non-Profits",
    "Politics": "Politics",

    # Lifestyle
    "Health and Fitness": "Health & Fitness",
    "Sports": "Sports",
    "Travel": "Travel",
    "Dating": "Dating",
    "Jobs/Employment": "Jobs & Careers",
    "Photo Sharing": "Photography",
    "Food": "Food & Recipes",

    # File & Storage
    "File Storage": "Cloud Storage",
    "P2P/File sharing": "File Sharing",

    # Other
    "Advertising": "Advertising",
    "Parked Domains": "Uncategorized",
    "URL Shorteners": "Utilities",
}

# Fallback keyword rules (used when OpenDNS has no category)
KEYWORD_RULES = {
    "Software & Technology": [
        "github.com", "gitlab.com", "stackoverflow.com", "developer.",
        "docs.", "api.", "devops", "kubernetes", "docker"
    ],
    "AI & Machine Learning": [
        "openai", "anthropic", "claude", "huggingface", "chatgpt",
        "machine learning", "neural", "llm"
    ],
    "Video & Streaming": [
        "youtube.com", "vimeo.com", "twitch.tv", "netflix", "hulu"
    ],
    "Gaming": [
        "dnd", "dndbeyond", "steam", "playstation", "xbox", "gaming"
    ],
    "Jobs & Careers": [
        "jobs.", "career", "linkedin.com/jobs", "indeed", "glassdoor"
    ],
    "Reference": [
        "wikipedia", "wiki", "documentation", "manual", "tutorial"
    ],
    "Shopping": [
        "amazon.com", "ebay.com", "etsy.com", "shopify"
    ],
    "News": [
        "news", "cnn.com", "bbc.com", "nytimes.com", "reuters"
    ],
    "Home & IoT": [
        "pi-hole", "raspberry", "solar", "powerwall", "home assistant"
    ],
}

# Domains/patterns to flag as potentially stale (internal/work)
STALE_PATTERNS = [
    r"\.internal\.",
    r"\.lan\.",
    r"localhost",
    r"127\.0\.0\.1",
    r"192\.168\.",
    r"10\.\d+\.",
]

# Default browser bookmarks to remove
BROWSER_DEFAULTS = [
    "Get Help",
    "Customize Firefox",
    "Get Involved",
    "About Us",
    "Help and Tutorials",
    "Chrome Web Store",
]


@dataclass
class Bookmark:
    title: str
    uri: str
    path: str
    guid: str
    date_added: int = 0
    last_modified: int = 0
    is_duplicate: bool = False
    is_dead: bool = False
    is_stale: bool = False
    is_browser_default: bool = False
    suggested_category: str = "Uncategorized"
    http_status: int = 0
    error_message: str = ""
    domain: str = ""


@dataclass
class DomainCategory:
    domain: str
    categories: list = field(default_factory=list)
    source: str = "unknown"  # "opendns", "keyword", "none"


def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def load_category_cache() -> dict[str, DomainCategory]:
    """Load cached domain categories."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                data = json.load(f)
                return {
                    domain: DomainCategory(
                        domain=domain,
                        categories=info.get("categories", []),
                        source=info.get("source", "unknown")
                    )
                    for domain, info in data.items()
                }
        except (json.JSONDecodeError, KeyError):
            pass
    return {}


def save_category_cache(cache: dict[str, DomainCategory]):
    """Save domain categories to cache."""
    data = {
        domain: {
            "categories": cat.categories,
            "source": cat.source
        }
        for domain, cat in cache.items()
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(data, f, indent=2)


async def lookup_opendns_category(client: httpx.AsyncClient, domain: str, semaphore: asyncio.Semaphore) -> DomainCategory:
    """Look up domain category from OpenDNS."""
    async with semaphore:
        try:
            url = f"https://domain.opendns.com/{domain}"
            response = await client.get(url, timeout=15.0, follow_redirects=True)

            if response.status_code == 200:
                html = response.text
                categories = []

                # Primary method: Look for "Tagged:" section with categories
                # Format: <h3>Tagged: <span class="normal">Category1, Category2</span></h3>
                tagged_pattern = r'<h3>Tagged:\s*<span[^>]*class="normal"[^>]*>\s*([^<]+)\s*</span>'
                match = re.search(tagged_pattern, html, re.IGNORECASE | re.DOTALL)
                if match:
                    # Split by comma and clean up
                    cats = match.group(1).strip()
                    for cat in cats.split(','):
                        cat = cat.strip()
                        if cat and len(cat) > 2:
                            categories.append(cat)

                # Fallback: Look for any span.normal after "Tagged"
                if not categories:
                    fallback_pattern = r'Tagged:.*?<span[^>]*>\s*([^<]+)\s*</span>'
                    match = re.search(fallback_pattern, html, re.IGNORECASE | re.DOTALL)
                    if match:
                        cats = match.group(1).strip()
                        for cat in cats.split(','):
                            cat = cat.strip()
                            if cat and len(cat) > 2:
                                categories.append(cat)

                # Clean up categories
                categories = list(set(cat.strip() for cat in categories if cat.strip()))

                if categories:
                    return DomainCategory(domain=domain, categories=categories, source="opendns")

            return DomainCategory(domain=domain, categories=[], source="opendns")

        except Exception:
            return DomainCategory(domain=domain, categories=[], source="error")


async def lookup_all_categories(domains: list[str], cache: dict[str, DomainCategory]) -> dict[str, DomainCategory]:
    """Look up categories for all domains, using cache when available."""
    # Filter out domains already in cache
    uncached = [d for d in domains if d not in cache and d]

    if not uncached:
        console.print("[dim]All domains found in cache[/dim]")
        return cache

    console.print(f"\n[bold]Looking up categories for {len(uncached)} domains...[/bold]")
    console.print(f"[dim]({len(domains) - len(uncached)} already cached)[/dim]")

    # Rate limit: 3 concurrent requests
    semaphore = asyncio.Semaphore(3)

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"},
        timeout=15.0
    ) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Querying OpenDNS...", total=len(uncached))

            async def lookup_with_progress(domain: str):
                result = await lookup_opendns_category(client, domain, semaphore)
                # Small delay to be polite to OpenDNS
                await asyncio.sleep(0.3)
                progress.advance(task)
                return result

            results = await asyncio.gather(*[lookup_with_progress(d) for d in uncached])

    # Update cache with results
    for result in results:
        cache[result.domain] = result

    # Save updated cache
    save_category_cache(cache)

    # Report stats
    found = len([r for r in results if r.categories])
    console.print(f"[green]Found categories for {found}/{len(uncached)} new domains[/green]")

    return cache


def map_opendns_to_folder(categories: list[str]) -> str:
    """Map OpenDNS categories to a folder name."""
    for cat in categories:
        if cat in OPENDNS_CATEGORY_MAP:
            return OPENDNS_CATEGORY_MAP[cat]

    # If no direct mapping, return the first category as-is
    if categories:
        return categories[0]

    return ""


def categorize_by_keywords(url: str, title: str, path: str) -> str:
    """Fallback categorization using keyword matching."""
    combined = f"{url.lower()} {title.lower()} {path.lower()}"

    for category, keywords in KEYWORD_RULES.items():
        for keyword in keywords:
            if keyword.lower() in combined:
                return category

    return "Uncategorized"


def detect_format(filepath: Path) -> BrowserFormat:
    """Detect the bookmark file format."""
    suffix = filepath.suffix.lower()

    if suffix == ".html" or suffix == ".htm":
        return BrowserFormat.FIREFOX_HTML

    if suffix == ".json":
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if "roots" in data and "bookmark_bar" in data.get("roots", {}):
                return BrowserFormat.CHROME_JSON

            if data.get("type") == "text/x-moz-place-container":
                return BrowserFormat.FIREFOX_JSON
            if data.get("root") == "placesRoot":
                return BrowserFormat.FIREFOX_JSON

        except (json.JSONDecodeError, KeyError):
            pass

    return BrowserFormat.UNKNOWN


def load_json(filepath: Path) -> dict:
    """Load JSON bookmark file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_firefox_html(filepath: Path) -> list[Bookmark]:
    """Parse Firefox HTML bookmark export."""
    import html.parser

    bookmarks = []
    current_path = []

    class BookmarkHTMLParser(html.parser.HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_h3 = False
            self.in_a = False
            self.current_title = ""
            self.current_href = ""
            self.current_add_date = 0
            self.current_last_modified = 0

        def handle_starttag(self, tag, attrs):
            attrs_dict = dict(attrs)
            if tag == "h3":
                self.in_h3 = True
                self.current_title = ""
            elif tag == "a":
                self.in_a = True
                self.current_title = ""
                self.current_href = attrs_dict.get("href", "")
                self.current_add_date = int(attrs_dict.get("add_date", 0) or 0)
                self.current_last_modified = int(attrs_dict.get("last_modified", 0) or 0)
            elif tag == "dl":
                pass

        def handle_endtag(self, tag):
            if tag == "h3":
                self.in_h3 = False
                if self.current_title:
                    current_path.append(self.current_title)
            elif tag == "a":
                self.in_a = False
                if self.current_href:
                    path = "/".join(current_path)
                    domain = extract_domain(self.current_href)
                    bookmark = Bookmark(
                        title=self.current_title,
                        uri=self.current_href,
                        path=path,
                        guid=f"html_{len(bookmarks)}",
                        date_added=self.current_add_date * 1000000,
                        last_modified=self.current_last_modified * 1000000,
                        domain=domain,
                    )
                    process_bookmark_flags(bookmark)
                    bookmarks.append(bookmark)
            elif tag == "dl":
                if current_path:
                    current_path.pop()

        def handle_data(self, data):
            if self.in_h3 or self.in_a:
                self.current_title += data.strip()

    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    parser = BookmarkHTMLParser()
    parser.feed(content)

    return bookmarks


def extract_firefox_bookmarks(node: dict, path: str = "") -> list[Bookmark]:
    """Recursively extract bookmarks from Firefox JSON."""
    bookmarks = []

    if node.get("type") == "text/x-moz-place":
        uri = node.get("uri", "")
        title = node.get("title", "")

        if uri.startswith("place:"):
            return bookmarks

        domain = extract_domain(uri)
        bookmark = Bookmark(
            title=title,
            uri=uri,
            path=path,
            guid=node.get("guid", ""),
            date_added=node.get("dateAdded", 0),
            last_modified=node.get("lastModified", 0),
            domain=domain,
        )
        process_bookmark_flags(bookmark)
        bookmarks.append(bookmark)

    elif node.get("type") == "text/x-moz-place-container":
        folder_name = node.get("title", "")
        new_path = f"{path}/{folder_name}" if path else folder_name

        for child in node.get("children", []):
            bookmarks.extend(extract_firefox_bookmarks(child, new_path))

    return bookmarks


def extract_chrome_bookmarks(node: dict, path: str = "") -> list[Bookmark]:
    """Recursively extract bookmarks from Chrome JSON."""
    bookmarks = []

    node_type = node.get("type", "")
    name = node.get("name", "")

    if node_type == "url":
        url = node.get("url", "")

        def chrome_time_to_unix_micro(chrome_time: str) -> int:
            try:
                ct = int(chrome_time)
                unix_micro = ct - (11644473600 * 1000000)
                return max(0, unix_micro)
            except (ValueError, TypeError):
                return 0

        domain = extract_domain(url)
        bookmark = Bookmark(
            title=name,
            uri=url,
            path=path,
            guid=node.get("guid", node.get("id", "")),
            date_added=chrome_time_to_unix_micro(node.get("date_added", "0")),
            last_modified=chrome_time_to_unix_micro(node.get("date_modified", "0")),
            domain=domain,
        )
        process_bookmark_flags(bookmark)
        bookmarks.append(bookmark)

    elif node_type == "folder":
        new_path = f"{path}/{name}" if path else name
        for child in node.get("children", []):
            bookmarks.extend(extract_chrome_bookmarks(child, new_path))

    return bookmarks


def process_bookmark_flags(bookmark: Bookmark):
    """Process a bookmark - check for defaults and stale URLs."""
    # Check if it's a browser default
    for default in BROWSER_DEFAULTS:
        if default.lower() in bookmark.title.lower():
            if any(d in bookmark.uri for d in ["mozilla.org", "google.com/chrome", "chrome.google.com"]):
                bookmark.is_browser_default = True
                break

    # Check if potentially stale (internal/work URL)
    for pattern in STALE_PATTERNS:
        if re.search(pattern, bookmark.uri, re.IGNORECASE):
            bookmark.is_stale = True
            break


def apply_categories(bookmarks: list[Bookmark], category_cache: dict[str, DomainCategory]):
    """Apply categories to bookmarks using OpenDNS data and keyword fallback."""
    opendns_count = 0
    keyword_count = 0
    uncategorized_count = 0

    for bookmark in bookmarks:
        category = "Uncategorized"

        # Try OpenDNS category first
        if bookmark.domain and bookmark.domain in category_cache:
            domain_cat = category_cache[bookmark.domain]
            if domain_cat.categories:
                category = map_opendns_to_folder(domain_cat.categories)
                if category and category != "Uncategorized":
                    opendns_count += 1
                    bookmark.suggested_category = category
                    continue

        # Fall back to keyword matching
        category = categorize_by_keywords(bookmark.uri, bookmark.title, bookmark.path)
        if category != "Uncategorized":
            keyword_count += 1
        else:
            uncategorized_count += 1

        bookmark.suggested_category = category

    console.print(f"\n[bold]Categorization results:[/bold]")
    console.print(f"  • OpenDNS: [green]{opendns_count}[/green] bookmarks")
    console.print(f"  • Keywords: [yellow]{keyword_count}[/yellow] bookmarks")
    console.print(f"  • Uncategorized: [red]{uncategorized_count}[/red] bookmarks")


def find_duplicates(bookmarks: list[Bookmark]) -> dict[str, list[Bookmark]]:
    """Find bookmarks with duplicate URLs."""
    url_map: dict[str, list[Bookmark]] = {}

    for bookmark in bookmarks:
        uri = bookmark.uri.strip().rstrip('/')
        if uri:
            if uri not in url_map:
                url_map[uri] = []
            url_map[uri].append(bookmark)

    return {uri: bms for uri, bms in url_map.items() if len(bms) > 1}


def mark_duplicates(bookmarks: list[Bookmark], duplicates: dict[str, list[Bookmark]]) -> list[Bookmark]:
    """Mark duplicate bookmarks, keeping the first occurrence."""
    for uri, dups in duplicates.items():
        sorted_dups = sorted(dups, key=lambda b: b.date_added)
        for dup in sorted_dups[1:]:
            for bookmark in bookmarks:
                if bookmark.guid == dup.guid:
                    bookmark.is_duplicate = True
                    break
    return bookmarks


async def check_link(client: httpx.AsyncClient, bookmark: Bookmark, semaphore: asyncio.Semaphore) -> Bookmark:
    """Check if a bookmark URL is still valid."""
    async with semaphore:
        if not bookmark.uri.startswith(('http://', 'https://')):
            bookmark.http_status = -1
            bookmark.error_message = "Non-HTTP URL"
            return bookmark

        if bookmark.is_stale:
            bookmark.http_status = -2
            bookmark.error_message = "Skipped (internal URL)"
            return bookmark

        try:
            response = await client.head(bookmark.uri, follow_redirects=True, timeout=10.0)
            bookmark.http_status = response.status_code
            if response.status_code >= 400:
                bookmark.is_dead = True
                bookmark.error_message = f"HTTP {response.status_code}"
        except httpx.TimeoutException:
            bookmark.http_status = 0
            bookmark.error_message = "Timeout"
            bookmark.is_dead = True
        except httpx.ConnectError:
            bookmark.http_status = 0
            bookmark.error_message = "Connection failed"
            bookmark.is_dead = True
        except Exception as e:
            bookmark.http_status = 0
            bookmark.error_message = str(e)[:50]
            bookmark.is_dead = True

        return bookmark


async def check_all_links(bookmarks: list[Bookmark], max_concurrent: int = 20) -> list[Bookmark]:
    """Check all bookmark links concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    to_check = [b for b in bookmarks if b.uri.startswith(('http://', 'https://')) and not b.is_stale]

    console.print(f"\n[bold]Checking {len(to_check)} links...[/bold] (skipping {len(bookmarks) - len(to_check)} internal/non-http)")

    async with httpx.AsyncClient(
        follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (compatible; bklint/1.0)"}
    ) as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Checking links...", total=len(to_check))

            async def check_with_progress(bookmark):
                result = await check_link(client, bookmark, semaphore)
                progress.advance(task)
                return result

            results = await asyncio.gather(*[check_with_progress(b) for b in to_check])

    checked_map = {b.guid: b for b in results}
    for bookmark in bookmarks:
        if bookmark.guid in checked_map:
            checked = checked_map[bookmark.guid]
            bookmark.http_status = checked.http_status
            bookmark.error_message = checked.error_message
            bookmark.is_dead = checked.is_dead

    return bookmarks


def get_clean_bookmarks(bookmarks: list[Bookmark], include_stale: bool = False) -> tuple[dict[str, list[Bookmark]], list[Bookmark]]:
    """Get bookmarks organized by category, excluding removed ones."""
    categories: dict[str, list[Bookmark]] = {}
    stale_bookmarks: list[Bookmark] = []

    for bookmark in bookmarks:
        if bookmark.is_duplicate or bookmark.is_dead or bookmark.is_browser_default:
            continue

        if bookmark.is_stale:
            if include_stale:
                stale_bookmarks.append(bookmark)
            continue

        category = bookmark.suggested_category
        if category not in categories:
            categories[category] = []
        categories[category].append(bookmark)

    return categories, stale_bookmarks


def build_firefox_structure(categories: dict[str, list[Bookmark]], stale_bookmarks: list[Bookmark]) -> dict:
    """Build Firefox JSON bookmark structure."""
    now = int(datetime.now().timestamp() * 1000000)

    def make_folder(title: str, children: list) -> dict:
        return {
            "guid": f"cleaned_{title.lower().replace(' ', '_').replace('/', '_').replace('&', 'and')}",
            "title": title,
            "index": 0,
            "dateAdded": now,
            "lastModified": now,
            "typeCode": 2,
            "type": "text/x-moz-place-container",
            "children": children
        }

    def make_bookmark(b: Bookmark) -> dict:
        return {
            "guid": b.guid,
            "title": b.title,
            "index": 0,
            "dateAdded": b.date_added,
            "lastModified": b.last_modified,
            "typeCode": 1,
            "type": "text/x-moz-place",
            "uri": b.uri
        }

    category_folders = []
    for category in sorted(categories.keys()):
        bms = categories[category]
        children = [make_bookmark(b) for b in sorted(bms, key=lambda x: x.title.lower())]
        category_folders.append(make_folder(category, children))

    if stale_bookmarks:
        children = [make_bookmark(b) for b in sorted(stale_bookmarks, key=lambda x: x.title.lower())]
        category_folders.append(make_folder("Archive (Internal)", children))

    return {
        "guid": "root________",
        "title": "",
        "index": 0,
        "dateAdded": now,
        "lastModified": now,
        "typeCode": 2,
        "type": "text/x-moz-place-container",
        "root": "placesRoot",
        "children": [
            {"guid": "menu________", "title": "menu", "index": 0, "dateAdded": now,
             "lastModified": now, "typeCode": 2, "type": "text/x-moz-place-container",
             "root": "bookmarksMenuFolder", "children": []},
            {"guid": "toolbar_____", "title": "toolbar", "index": 1, "dateAdded": now,
             "lastModified": now, "typeCode": 2, "type": "text/x-moz-place-container",
             "root": "bookmarksToolbarFolder", "children": category_folders},
            {"guid": "unfiled_____", "title": "unfiled", "index": 2, "dateAdded": now,
             "lastModified": now, "typeCode": 2, "type": "text/x-moz-place-container",
             "root": "unfiledBookmarksFolder", "children": []},
            {"guid": "mobile______", "title": "mobile", "index": 3, "dateAdded": now,
             "lastModified": now, "typeCode": 2, "type": "text/x-moz-place-container",
             "root": "mobileFolder", "children": []}
        ]
    }


def build_chrome_structure(categories: dict[str, list[Bookmark]], stale_bookmarks: list[Bookmark]) -> dict:
    """Build Chrome JSON bookmark structure."""
    def unix_micro_to_chrome_time(unix_micro: int) -> str:
        if unix_micro <= 0:
            unix_micro = int(datetime.now().timestamp() * 1000000)
        chrome_time = unix_micro + (11644473600 * 1000000)
        return str(chrome_time)

    def make_folder(title: str, children: list, folder_id: int) -> dict:
        return {
            "children": children,
            "date_added": unix_micro_to_chrome_time(0),
            "date_modified": unix_micro_to_chrome_time(0),
            "guid": f"cleaned_{folder_id}",
            "id": str(folder_id),
            "name": title,
            "type": "folder"
        }

    def make_bookmark(b: Bookmark, bm_id: int) -> dict:
        return {
            "date_added": unix_micro_to_chrome_time(b.date_added),
            "date_last_used": "0",
            "guid": b.guid,
            "id": str(bm_id),
            "name": b.title,
            "type": "url",
            "url": b.uri
        }

    bookmark_bar_children = []
    current_id = 10

    for category in sorted(categories.keys()):
        bms = categories[category]
        folder_children = []
        for b in sorted(bms, key=lambda x: x.title.lower()):
            folder_children.append(make_bookmark(b, current_id))
            current_id += 1
        bookmark_bar_children.append(make_folder(category, folder_children, current_id))
        current_id += 1

    if stale_bookmarks:
        folder_children = []
        for b in sorted(stale_bookmarks, key=lambda x: x.title.lower()):
            folder_children.append(make_bookmark(b, current_id))
            current_id += 1
        bookmark_bar_children.append(make_folder("Archive (Internal)", folder_children, current_id))

    return {
        "checksum": "",
        "roots": {
            "bookmark_bar": {
                "children": bookmark_bar_children,
                "date_added": unix_micro_to_chrome_time(0),
                "date_modified": unix_micro_to_chrome_time(0),
                "guid": "bookmark_bar",
                "id": "1",
                "name": "Bookmarks bar",
                "type": "folder"
            },
            "other": {
                "children": [],
                "date_added": unix_micro_to_chrome_time(0),
                "date_modified": unix_micro_to_chrome_time(0),
                "guid": "other",
                "id": "2",
                "name": "Other bookmarks",
                "type": "folder"
            },
            "synced": {
                "children": [],
                "date_added": unix_micro_to_chrome_time(0),
                "date_modified": unix_micro_to_chrome_time(0),
                "guid": "synced",
                "id": "3",
                "name": "Mobile bookmarks",
                "type": "folder"
            }
        },
        "version": 1
    }


def save_firefox_html(structure: dict, output_path: Path):
    """Save bookmarks as Firefox HTML format."""
    import html as html_module

    def timestamp_to_unix(ts: int) -> int:
        if ts > 0:
            return ts // 1000000
        return int(datetime.now().timestamp())

    lines = [
        '<!DOCTYPE NETSCAPE-Bookmark-file-1>',
        '<!-- This is an automatically generated file.',
        '     It will be read and overwritten.',
        '     DO NOT EDIT! -->',
        '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">',
        '<TITLE>Bookmarks</TITLE>',
        '<H1>Bookmarks Menu</H1>',
        '',
        '<DL><p>'
    ]

    def process_node(node: dict, indent: int = 1):
        indent_str = '    ' * indent

        if node.get("type") == "text/x-moz-place":
            title = html_module.escape(node.get("title", ""))
            uri = html_module.escape(node.get("uri", ""))
            add_date = timestamp_to_unix(node.get("dateAdded", 0))
            last_modified = timestamp_to_unix(node.get("lastModified", 0))
            lines.append(f'{indent_str}<DT><A HREF="{uri}" ADD_DATE="{add_date}" LAST_MODIFIED="{last_modified}">{title}</A>')

        elif node.get("type") == "text/x-moz-place-container":
            title = html_module.escape(node.get("title", ""))
            add_date = timestamp_to_unix(node.get("dateAdded", 0))
            last_modified = timestamp_to_unix(node.get("lastModified", 0))

            if node.get("root") == "placesRoot":
                for child in node.get("children", []):
                    process_node(child, indent)
                return

            if node.get("root") == "bookmarksToolbarFolder":
                lines.append(f'{indent_str}<DT><H3 ADD_DATE="{add_date}" LAST_MODIFIED="{last_modified}" PERSONAL_TOOLBAR_FOLDER="true">{title or "Bookmarks Toolbar"}</H3>')
            elif node.get("root") == "unfiledBookmarksFolder":
                lines.append(f'{indent_str}<DT><H3 ADD_DATE="{add_date}" LAST_MODIFIED="{last_modified}" UNFILED_BOOKMARKS_FOLDER="true">{title or "Other Bookmarks"}</H3>')
            elif title:
                lines.append(f'{indent_str}<DT><H3 ADD_DATE="{add_date}" LAST_MODIFIED="{last_modified}">{title}</H3>')
            else:
                for child in node.get("children", []):
                    process_node(child, indent)
                return

            lines.append(f'{indent_str}<DL><p>')
            for child in node.get("children", []):
                process_node(child, indent + 1)
            lines.append(f'{indent_str}</DL><p>')

    process_node(structure)
    lines.append('</DL><p>')

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    console.print(f"[bold green]✓[/bold green] Saved Firefox HTML: {output_path}")


def save_json(data: dict, output_path: Path, browser: str):
    """Save bookmarks as JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    console.print(f"[bold green]✓[/bold green] Saved {browser} JSON: {output_path}")


def save_report(bookmarks: list[Bookmark], duplicates: dict, category_cache: dict[str, DomainCategory], output_path: Path):
    """Save a detailed report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("BKLINT - BOOKMARK CLEANUP REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("DUPLICATES REMOVED:\n")
        for uri, dups in duplicates.items():
            f.write(f"\nURL: {uri}\n")
            for d in dups:
                status = "REMOVED" if d.is_duplicate else "KEPT"
                f.write(f"  [{status}] {d.title} (in {d.path})\n")

        f.write("\n\nDEAD LINKS REMOVED:\n")
        for b in bookmarks:
            if b.is_dead:
                f.write(f"  • {b.title}\n    URL: {b.uri}\n    Error: {b.error_message}\n\n")

        f.write("\n\nINTERNAL URLs (kept in Archive):\n")
        for b in bookmarks:
            if b.is_stale and not b.is_duplicate:
                f.write(f"  • {b.title}\n    URL: {b.uri}\n\n")

        f.write("\n\nCATEGORY ASSIGNMENTS:\n")
        by_category: dict[str, list[Bookmark]] = {}
        for b in bookmarks:
            if not b.is_duplicate and not b.is_dead:
                cat = b.suggested_category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(b)

        for cat in sorted(by_category.keys()):
            f.write(f"\n{cat} ({len(by_category[cat])} bookmarks):\n")
            for b in sorted(by_category[cat], key=lambda x: x.title.lower())[:10]:
                source = ""
                if b.domain in category_cache:
                    dc = category_cache[b.domain]
                    if dc.categories:
                        source = f" [OpenDNS: {', '.join(dc.categories[:2])}]"
                f.write(f"  • {b.title[:50]}{source}\n")
            if len(by_category[cat]) > 10:
                f.write(f"  ... and {len(by_category[cat]) - 10} more\n")

    console.print(f"[bold green]✓[/bold green] Saved report: {output_path}")


def print_report(bookmarks: list[Bookmark], duplicates: dict[str, list[Bookmark]]):
    """Print analysis report."""
    console.print("\n[bold blue]═══ Bookmark Analysis Report ═══[/bold blue]\n")

    total = len(bookmarks)
    dead = len([b for b in bookmarks if b.is_dead])
    stale = len([b for b in bookmarks if b.is_stale])
    browser_default = len([b for b in bookmarks if b.is_browser_default])
    dup_count = sum(len(v) - 1 for v in duplicates.values())

    table = Table(title="Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right", style="green")

    table.add_row("Total bookmarks", str(total))
    table.add_row("Duplicate entries", str(dup_count))
    table.add_row("Dead/broken links", str(dead))
    table.add_row("Internal/work URLs", str(stale))
    table.add_row("Browser defaults", str(browser_default))
    table.add_row("[bold]To be removed[/bold]", f"[bold red]{dup_count + dead + browser_default}[/bold red]")

    console.print(table)

    if duplicates:
        console.print("\n[bold yellow]Duplicate URLs:[/bold yellow]")
        for uri, dups in list(duplicates.items())[:10]:
            console.print(f"  [dim]{uri[:70]}{'...' if len(uri) > 70 else ''}[/dim]")
            for d in dups:
                status = "[red](removing)[/red]" if d.is_duplicate else "[green](keeping)[/green]"
                console.print(f"    • {d.title[:50]} {status}")

    dead_bookmarks = [b for b in bookmarks if b.is_dead]
    if dead_bookmarks:
        console.print(f"\n[bold red]Dead/Broken Links ({len(dead_bookmarks)}):[/bold red]")
        for b in dead_bookmarks[:15]:
            console.print(f"  • {b.title[:40]} - [dim]{b.error_message}[/dim]")
        if len(dead_bookmarks) > 15:
            console.print(f"  [dim]... and {len(dead_bookmarks) - 15} more[/dim]")

    console.print("\n[bold green]Categories:[/bold green]")
    categories: dict[str, int] = {}
    for b in bookmarks:
        if not b.is_duplicate and not b.is_dead and not b.is_browser_default:
            cat = b.suggested_category
            categories[cat] = categories.get(cat, 0) + 1

    for cat in sorted(categories.keys()):
        console.print(f"  • {cat}: {categories[cat]} bookmarks")


def find_bookmark_file() -> Path | None:
    """Find a bookmark file in the current directory."""
    for pattern in ["*.json", "*.html", "*.htm"]:
        for f in Path(".").glob(pattern):
            if "cleaned" in f.name or "chrome" in f.name or "firefox" in f.name:
                continue  # Skip output files
            fmt = detect_format(f)
            if fmt != BrowserFormat.UNKNOWN:
                return f
    return None


def parse_args():
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="bklint",
        description="Bookmark Linter - Clean up and organize Firefox & Chrome bookmarks",
        epilog="Examples:\n"
               "  bklint -i bookmarks.json           Process a specific file\n"
               "  bklint -i bookmarks.html -y        Run non-interactively\n"
               "  bklint                             Interactive mode (auto-finds files)\n",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "-i", "--input",
        type=Path,
        metavar="FILE",
        help="Input bookmark file (Firefox JSON/HTML or Chrome JSON)"
    )

    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Yes to all prompts (non-interactive mode)"
    )

    parser.add_argument(
        "--no-dead-links",
        action="store_true",
        help="Skip dead link checking"
    )

    parser.add_argument(
        "--no-opendns",
        action="store_true",
        help="Skip OpenDNS category lookup"
    )

    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Don't include internal/work URLs in archive"
    )

    parser.add_argument(
        "-o", "--output",
        choices=["firefox", "chrome", "both"],
        default="both",
        help="Output format (default: both)"
    )

    return parser.parse_args()


@dataclass
class Options:
    """Runtime options collected from user or CLI."""
    input_file: Path
    check_dead_links: bool = True
    lookup_opendns: bool = True
    include_archive: bool = True
    output_firefox: bool = True
    output_chrome: bool = True


def collect_options_interactive(args) -> Options | None:
    """Collect all options interactively before processing."""
    console.print("[bold blue]bklint - Bookmark Linter[/bold blue]")
    console.print("[dim]Clean up and organize Firefox & Chrome bookmarks[/dim]\n")

    # 1. Find or prompt for input file
    input_file = args.input
    if not input_file:
        input_file = find_bookmark_file()
        if input_file:
            if not Confirm.ask(f"Found [bold]{input_file}[/bold]. Use this file?", default=True):
                input_file = None

    if not input_file:
        file_path = Prompt.ask("Enter bookmark file path")
        if not file_path:
            console.print("[red]No file specified. Exiting.[/red]")
            return None
        input_file = Path(file_path)

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        return None

    # Validate format
    fmt = detect_format(input_file)
    if fmt == BrowserFormat.UNKNOWN:
        console.print("[red]Error: Unknown bookmark format. Supports Firefox (JSON/HTML) and Chrome (JSON).[/red]")
        return None

    format_names = {
        BrowserFormat.FIREFOX_JSON: "Firefox JSON",
        BrowserFormat.FIREFOX_HTML: "Firefox HTML",
        BrowserFormat.CHROME_JSON: "Chrome JSON",
    }
    console.print(f"Detected format: [bold]{format_names[fmt]}[/bold]\n")

    # 2. Collect all options upfront
    console.print("[bold]Configuration:[/bold]")

    check_dead_links = Confirm.ask(
        "  Check for dead/broken links? (takes a few minutes)",
        default=True
    )

    lookup_opendns = Confirm.ask(
        "  Look up domain categories from OpenDNS?",
        default=True
    )

    include_archive = Confirm.ask(
        "  Include internal/work URLs in an 'Archive' folder?",
        default=True
    )

    # 3. Output format selection
    console.print("\n[bold]Save results as:[/bold]")
    console.print("  1. Firefox only (HTML + JSON)")
    console.print("  2. Chrome only (JSON)")
    console.print("  3. Both Firefox and Chrome")
    console.print("  4. Don't save (dry run)")

    output_choice = Prompt.ask("Select", choices=["1", "2", "3", "4"], default="3")

    output_firefox = output_choice in ["1", "3"]
    output_chrome = output_choice in ["2", "3"]

    if output_choice == "4":
        console.print("\n[yellow]Dry run mode - no files will be saved[/yellow]")

    return Options(
        input_file=input_file,
        check_dead_links=check_dead_links,
        lookup_opendns=lookup_opendns,
        include_archive=include_archive,
        output_firefox=output_firefox,
        output_chrome=output_chrome,
    )


def get_options_from_args(args) -> Options | None:
    """Get options from command line args (non-interactive mode)."""
    console.print("[bold blue]bklint - Bookmark Linter[/bold blue]")
    console.print("[dim]Running in non-interactive mode (-y)[/dim]\n")

    input_file = args.input
    if not input_file:
        input_file = find_bookmark_file()
        if not input_file:
            console.print("[red]Error: No bookmark file found. Use -i to specify one.[/red]")
            return None

    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        return None

    fmt = detect_format(input_file)
    if fmt == BrowserFormat.UNKNOWN:
        console.print("[red]Error: Unknown bookmark format.[/red]")
        return None

    format_names = {
        BrowserFormat.FIREFOX_JSON: "Firefox JSON",
        BrowserFormat.FIREFOX_HTML: "Firefox HTML",
        BrowserFormat.CHROME_JSON: "Chrome JSON",
    }
    console.print(f"Input: [bold]{input_file}[/bold] ({format_names[fmt]})")

    return Options(
        input_file=input_file,
        check_dead_links=not args.no_dead_links,
        lookup_opendns=not args.no_opendns,
        include_archive=not args.no_archive,
        output_firefox=args.output in ["firefox", "both"],
        output_chrome=args.output in ["chrome", "both"],
    )


async def run_with_options(opts: Options):
    """Run the linting process with the given options."""
    input_file = opts.input_file
    fmt = detect_format(input_file)

    # Load and parse bookmarks
    console.print(f"\n[bold]Processing {input_file}...[/bold]")

    if fmt == BrowserFormat.FIREFOX_JSON:
        data = load_json(input_file)
        bookmarks = extract_firefox_bookmarks(data)
    elif fmt == BrowserFormat.FIREFOX_HTML:
        bookmarks = parse_firefox_html(input_file)
    elif fmt == BrowserFormat.CHROME_JSON:
        data = load_json(input_file)
        bookmarks = []
        for root_name in ["bookmark_bar", "other", "synced"]:
            if root_name in data.get("roots", {}):
                bookmarks.extend(extract_chrome_bookmarks(data["roots"][root_name]))
    else:
        return

    console.print(f"Found [bold]{len(bookmarks)}[/bold] bookmarks")

    # Find and mark duplicates
    console.print("Finding duplicates...")
    duplicates = find_duplicates(bookmarks)
    console.print(f"Found [bold]{len(duplicates)}[/bold] URLs with duplicates")
    bookmarks = mark_duplicates(bookmarks, duplicates)

    # Check for dead links
    if opts.check_dead_links:
        bookmarks = await check_all_links(bookmarks)

    # Extract unique domains for categorization
    unique_domains = list(set(b.domain for b in bookmarks if b.domain and not b.is_stale))
    console.print(f"\nFound [bold]{len(unique_domains)}[/bold] unique domains to categorize")

    # Load cache and lookup categories
    category_cache = load_category_cache()

    if opts.lookup_opendns:
        category_cache = await lookup_all_categories(unique_domains, category_cache)

    # Apply categories to bookmarks
    apply_categories(bookmarks, category_cache)

    # Print report
    print_report(bookmarks, duplicates)

    # Get clean bookmarks
    categories, stale_bookmarks = get_clean_bookmarks(bookmarks, opts.include_archive)

    # Count results
    final_count = sum(len(bms) for bms in categories.values()) + len(stale_bookmarks)
    removed = len(bookmarks) - final_count
    console.print(f"\n[bold]Final result:[/bold] {final_count} bookmarks (removed {removed})")

    # Save outputs
    if not opts.output_firefox and not opts.output_chrome:
        console.print("\n[yellow]Dry run complete - no files saved[/yellow]")
        return

    base_name = input_file.stem.replace("bookmarks", "").strip("-_ ") or "cleaned"

    if opts.output_firefox:
        firefox_structure = build_firefox_structure(categories, stale_bookmarks)
        save_firefox_html(firefox_structure, Path(f"bookmarks-{base_name}-firefox.html"))
        save_json(firefox_structure, Path(f"bookmarks-{base_name}-firefox.json"), "Firefox")

    if opts.output_chrome:
        chrome_structure = build_chrome_structure(categories, stale_bookmarks)
        save_json(chrome_structure, Path(f"bookmarks-{base_name}-chrome.json"), "Chrome")

    save_report(bookmarks, duplicates, category_cache, Path(f"bklint-report-{base_name}.txt"))

    console.print("\n[bold green]Done![/bold green]")

    if opts.output_firefox:
        console.print("\n[bold]To import into Firefox:[/bold]")
        console.print("  1. Open Library (Cmd+Shift+O / Ctrl+Shift+O)")
        console.print("  2. Import and Backup → Import Bookmarks from HTML...")
        console.print(f"  3. Select [bold]bookmarks-{base_name}-firefox.html[/bold]")

    if opts.output_chrome:
        console.print("\n[bold]To import into Chrome:[/bold]")
        console.print("  1. Go to chrome://bookmarks")
        console.print("  2. Click ⋮ menu → Import bookmarks")
        console.print("  3. Chrome imports HTML, so use the Firefox HTML file")


async def main_async():
    """Main async function."""
    args = parse_args()

    # Collect options based on mode
    if args.yes:
        opts = get_options_from_args(args)
    else:
        opts = collect_options_interactive(args)

    if opts is None:
        return

    # Run the linting process
    await run_with_options(opts)


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

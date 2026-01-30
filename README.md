# bklint - Bookmark Linter

A command-line tool to clean up and organize browser bookmarks from Firefox and Chrome.

![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)

![Firefox](https://img.shields.io/badge/Firefox-supported-FF7139?logo=firefox-browser&logoColor=white)
![Chrome](https://img.shields.io/badge/Chrome-supported-4285F4?logo=google-chrome&logoColor=white)


## Features

- **Auto-detect format** - Supports Firefox (JSON/HTML) and Chrome (JSON) bookmark exports
- **Remove duplicates** - Finds and removes duplicate URLs, keeping the oldest entry
- **Check dead links** - Validates URLs and removes broken/dead links (optional)
- **Smart categorization** - Uses [OpenDNS Domain Tagging](https://community.opendns.com/domaintagging/) for intelligent categorization
- **Reorganize structure** - Flattens nested folders into clean category-based organization
- **Multi-format export** - Outputs to both Firefox and Chrome formats
- **Caching** - Caches OpenDNS lookups for fast subsequent runs

## Installation

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
# Clone or download the project
cd bklint

# Install dependencies
uv sync
```

## Usage

### Basic Usage

```bash
# Run interactively (auto-finds bookmark files in current directory)
uv run python bklint.py

# Specify a bookmark file directly
uv run python bklint.py bookmarks.json
uv run python bklint.py bookmarks.html
```

### Export Bookmarks from Your Browser

**Firefox:**
1. Open Library: `Cmd+Shift+O` (Mac) or `Ctrl+Shift+O` (Windows/Linux)
2. Click "Import and Backup" → "Export Bookmarks to HTML..." or "Backup..."

**Chrome:**
1. Go to `chrome://bookmarks`
2. Click ⋮ menu → "Export bookmarks"

Or use Chrome's internal bookmark file directly:
```bash
# macOS
uv run python bklint.py ~/Library/Application\ Support/Google/Chrome/Default/Bookmarks

# Linux
uv run python bklint.py ~/.config/google-chrome/Default/Bookmarks

# Windows
uv run python bklint.py "%LOCALAPPDATA%\Google\Chrome\User Data\Default\Bookmarks"
```

### Command Examples

```bash
# Process Firefox JSON backup
uv run python bklint.py bookmarks-2024-01-15.json

# Process Firefox HTML export
uv run python bklint.py bookmarks.html

# Process Chrome bookmarks directly
uv run python bklint.py ~/Library/Application\ Support/Google/Chrome/Default/Bookmarks
```

## What It Does

### 1. Duplicate Removal
Finds URLs that appear multiple times and keeps only the oldest entry (by date added).

### 2. Dead Link Detection
Optionally checks each URL to verify it's still accessible. Removes links that return:
- HTTP 4xx/5xx errors
- Connection failures
- Timeouts

### 3. Smart Categorization

**OpenDNS Lookup (Primary):**
Queries OpenDNS's community-maintained domain database for accurate categorization. Categories include:
- Software/Technology
- News/Media
- Shopping/Ecommerce
- Video Sharing
- Social Networking
- Games
- And 70+ more

**Keyword Fallback:**
For domains not in OpenDNS, falls back to keyword matching based on URL and title.

### 4. Internal/Work URL Detection
Identifies internal URLs (localhost, private IPs, corporate domains) and optionally archives them separately.

### 5. Browser Default Removal
Removes default bookmarks added by Firefox and Chrome (Help pages, Get Involved, etc.).

## Output Files

After running, bklint creates:

| File | Description |
|------|-------------|
| `bookmarks-*-firefox.html` | Import into Firefox (and Chrome) |
| `bookmarks-*-firefox.json` | Firefox restore file |
| `bookmarks-*-chrome.json` | Chrome bookmark format |
| `bklint-report-*.txt` | Detailed report of changes |
| `.bklint-category-cache.json` | Cached OpenDNS lookups |

## Import Cleaned Bookmarks

**Firefox:**
1. Open Library: `Cmd+Shift+O` / `Ctrl+Shift+O`
2. Click "Import and Backup" → "Import Bookmarks from HTML..."
3. Select the `*-firefox.html` file

**Chrome:**
1. Go to `chrome://bookmarks`
2. Click ⋮ menu → "Import bookmarks"
3. Select the `*-firefox.html` file (Chrome imports HTML format)

## Configuration

### Custom Stale URL Patterns

Edit `STALE_PATTERNS` in `bklint.py` to add your own internal/work domains:

```python
STALE_PATTERNS = [
    r"\.yourcompany\.com",
    r"\.internal\.",
    r"localhost",
    # ... add more patterns
]
```

### Custom Keyword Rules

Edit `KEYWORD_RULES` to add fallback categorization rules:

```python
KEYWORD_RULES = {
    "Your Category": [
        "keyword1", "keyword2", "domain.com"
    ],
}
```

### OpenDNS Category Mapping

Edit `OPENDNS_CATEGORY_MAP` to customize how OpenDNS categories map to folder names:

```python
OPENDNS_CATEGORY_MAP = {
    "Software/Technology": "Dev Tools",  # Rename category
    "Video Sharing": "Videos",
}
```

## How Categorization Works

```
┌─────────────────────┐
│  Extract Domain     │
│  from URL           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Check Cache        │──── Hit ────▶ Use cached category
└──────────┬──────────┘
           │ Miss
           ▼
┌─────────────────────┐
│  Query OpenDNS      │──── Found ──▶ Map to folder name
│  domain.opendns.com │
└──────────┬──────────┘
           │ Not found
           ▼
┌─────────────────────┐
│  Keyword Matching   │──── Match ──▶ Use keyword category
└──────────┬──────────┘
           │ No match
           ▼
┌─────────────────────┐
│  "Uncategorized"    │
└─────────────────────┘
```

## License

MIT

## Credits

- Domain categorization powered by [OpenDNS Community Domain Tagging](https://community.opendns.com/domaintagging/)
- Built with [httpx](https://www.python-httpx.org/) and [rich](https://rich.readthedocs.io/)

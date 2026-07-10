# Leads University — Islamabad Campus Chatbot

A RAG (retrieval-augmented generation) chatbot for the Islamabad Campus of
Lahore Leads University (leads.edu.pk), built with FireCrawl (scraping),
Chroma (local vector store), FastAPI (backend), and OpenAI (generation).

## 1. Open the project in VS Code

1. Unzip this folder somewhere on your machine and open it in VS Code
   (`File > Open Folder...`).
2. Install the **Python extension** for VS Code if you don't have it
   (Extensions panel → search "Python" → the Microsoft one).
3. Open a terminal inside VS Code: `` Terminal > New Terminal `` (or
   `` Ctrl+` ``).

## 2. Create a virtual environment and install dependencies

In the VS Code terminal:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

VS Code will usually prompt you to select `.venv` as the interpreter for
the folder — click "Yes"/"Select Interpreter" when it does (bottom-right
corner also lets you pick it manually).

## 3. Get your API keys

**FireCrawl API key:**
1. Sign up at https://firecrawl.dev
2. Go to your Dashboard → API Keys, copy the key (starts with `fc-`)
3. The free tier includes 1,000 credits/month — each page scraped/crawled
   costs 1 credit, so that's plenty for a single-campus site.

**OpenAI API key:**
1. Sign up / log in at https://platform.openai.com
2. Go to API Keys → Create new secret key
3. Note: this requires billing to be set up on your OpenAI account (a few
   cents of usage will cover a lot of test queries with `gpt-4o-mini`)

Copy `.env.example` to `.env` and fill in both keys:

```bash
cp .env.example .env    # macOS/Linux
copy .env.example .env  # Windows
```

Then edit `.env` in VS Code and paste your real keys in.

## 4. Scrape the site with FireCrawl

```bash
python ingest/scrape.py
```

What this does:
- Starts crawling from `https://leads.edu.pk/islamabad-campus/`
- Follows internal links (up to 150 pages) and pulls back clean markdown
  for each page
- Keeps only pages that look Islamabad-relevant or are shared/general
  pages every campus should know about (About Us, Programs, Admissions,
  Faculty, Policies, Fees, Contact, etc.) — see `ISLAMABAD_URL_HINTS` and
  `SHARED_URL_HINTS` in `ingest/scrape.py` if you want to widen or narrow
  that
- Saves one JSON file per page into `data/raw/`

**Before moving on, skim a few files in `data/raw/`.** When I looked at
the live site, I found spam content (casino/gambling ad copy) injected
into otherwise normal page text — likely from a compromised plugin or
malicious script on the WordPress install. `build_index.py` filters
common spam keywords automatically, but it's worth a manual glance so
nothing embarrassing slips into the bot's answers, and worth reporting to
whoever manages the WordPress site.

## 5. Build the vector index

```bash
python ingest/build_index.py
```

This cleans each page (strips spam-flagged chunks and menu boilerplate),
splits the text into ~1000-character chunks, and stores them in a local
Chroma database at `data/chroma/`. Chroma's default embedding model runs
locally — no extra API key needed for this step.

Re-run this any time you re-scrape and want to refresh the bot's
knowledge; it rebuilds the collection from scratch each time.

## 6. Run the backend

```bash
uvicorn app.main:app --reload
```

This starts the API at `http://localhost:8000`. Open
`http://localhost:8000` in a browser — it serves the test chat widget
(`frontend/widget.html`) so you can try it immediately.

## 7. Embed it on the real site

Once you're happy with answers, either:
- Drop `frontend/widget.html`'s `<div id="chat">` + `<script>` block into
  a WordPress page/footer (as an HTML block or via a plugin like "Insert
  Headers and Footers"), pointing `API_URL` at your deployed backend, or
- Turn it into a floating chat bubble instead of a full-page widget (happy
  to help with that once you've confirmed the RAG pipeline gives good
  answers).

You'll need to deploy `app/` somewhere reachable from the public internet
(Railway, Render, a VPS, etc.) — `localhost:8000` only works on your own
machine.

## Project structure

```
leads-chatbot/
├── .env.example          # copy to .env and fill in API keys
├── requirements.txt
├── ingest/
│   ├── scrape.py          # FireCrawl crawl -> data/raw/*.json
│   └── build_index.py     # clean + chunk + embed -> data/chroma/
├── app/
│   ├── rag.py             # retrieval + Claude generation
│   └── main.py            # FastAPI /chat endpoint
├── frontend/
│   └── widget.html        # standalone test widget
└── data/                  # created at runtime (raw pages + vector db)
```

## Notes / things to adjust for production

- **CORS**: `app/main.py` currently allows all origins (`allow_origins=["*"]`).
  Lock this down to your actual WordPress domain before going live.
- **Model choice**: `app/rag.py` calls `gpt-4o-mini` — cheap and plenty
  capable for a campus FAQ bot. Swap in `gpt-4o` in the same spot if you
  want higher-quality answers and don't mind the extra cost.
- **Re-scraping cadence**: campus info (fee structures, admission dates)
  changes periodically — consider re-running steps 4–5 on a schedule
  (cron job, GitHub Action, etc.) rather than only once.
- **Spam filtering**: `SPAM_PATTERNS` in `ingest/build_index.py` is a
  starting point, not a guarantee — check `data/raw/` manually the first
  time and extend the pattern list if you spot other injected junk.

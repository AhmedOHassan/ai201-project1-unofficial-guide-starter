# The Unofficial Guide — Project 1

A small RAG system that answers plain-language questions about off-campus
student housing near NC State, grounded only in real r/NCSU review threads and
with the source threads cited on every answer.

## How to run it

```bash
python -m venv .venv
source .venv/Scripts/activate          # Windows (Git Bash); use .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
cp .env.example .env                    # then paste your Groq API key into .env

python app.py                           # launches the UI at http://localhost:7860
```

`app.py` is the only thing you need to run, on first launch it bootstraps the
pipeline automatically (cleans + chunks the threads into `data/chunks.json`, then
embeds them into ChromaDB), and skips that work on later launches once it's built.

You can also run the stages individually to inspect them:

```bash
python pipeline.py     # clean the 10 threads -> chunk -> data/chunks.json (prints stats + sample chunks)
python index.py        # embed chunks into ChromaDB + print retrieval on the eval queries
python evaluate.py     # run all 5 evaluation questions end-to-end
python -m pytest -q    # chunking, retrieval, and grounded-generation tests
```

Pipeline: **Ingestion → Chunking → Embedding + Vector Store → Retrieval → Generation**
(`pipeline.py` → `index.py` → `query.py` / `app.py`). The full design + diagram
lives in [planning.md](planning.md).

---

## Domain

My system covers **off-campus student housing near NC State**, apartment
reviews, hidden costs (parking, utilities, fees), neighborhood safety, and how
realistic the Wolfline bus access actually is for each complex.

This is worth doing because the official sources are useless for an honest
decision. Apartment websites use staged photos, leave parking and utility fees
off the headline rent, and their Google reviews are heavily padded by
management (students in my docs literally describe being offered free pizza for
5-star reviews). The information you actually need, which buildings have
windows that don't open, which management company tows your car, what a 4x4
really costs once you add everything up, only lives in scattered Reddit threads
on r/NCSU, spread across years of comments. A student researching this by hand
would have to read dozens of threads and cross-reference them, which is exactly
the kind of buried knowledge a RAG system is good at surfacing.

---

## Document Sources

All 10 sources are r/NCSU threads, saved as `.txt` files in [documents/](documents/).

| #   | Source (thread title)                                               | Type          | URL or file path                                                                               |
| --- | ------------------------------------------------------------------- | ------------- | ---------------------------------------------------------------------------------------------- |
| 1   | What off campus apartments do you recommend?                        | Reddit thread | https://www.reddit.com/r/NCSU/comments/1c4veee/what_off_campus_apartments_do_you_recommend_or/ |
| 2   | [Safety Alert] Prowler / Casing Houses near Avent Ferry & Socket Dr | Reddit thread | https://www.reddit.com/r/NCSU/comments/1trnl1b/safety_alert_prowler_casing_houses_near_avent/  |
| 3   | Off Campus Housing Advice Needed                                    | Reddit thread | https://www.reddit.com/r/NCSU/comments/12fsevy/off_campus_housing_advice_needed/               |
| 4   | NC State Apartment Living                                           | Reddit thread | https://www.reddit.com/r/NCSU/comments/1g87da8/nc_state_apartment_living/                      |
| 5   | Off Campus Housing Mega thread                                      | Reddit thread | https://www.reddit.com/r/NCSU/comments/brtjus/off_campus_housing/                              |
| 6   | Off Campus Housing: University Woods                                | Reddit thread | https://www.reddit.com/r/NCSU/comments/12bjdtd/off_campus_housing/                             |
| 7   | Opinions on "The Wilde" Apartments                                  | Reddit thread | https://www.reddit.com/r/NCSU/comments/12tonfj/opinions_on_the_wilde_apartments/               |
| 8   | Off-campus housing — Trinity properties                             | Reddit thread | https://www.reddit.com/r/NCSU/comments/w44ieh/offcampus_housing_trinity_properties/            |
| 9   | No openable windows at The Standard                                 | Reddit thread | https://www.reddit.com/r/NCSU/comments/1ts9070/no_openable_windows_at_the_standard/            |
| 10  | Valentine Commons reviews                                           | Reddit thread | https://www.reddit.com/r/NCSU/comments/1no6n56/valentine_commons_reviews/                      |

---

## Chunking Strategy

**Chunk size:** 240 tokens

**Overlap:** 40 tokens

**Why these choices fit your documents:** These threads are conversational, not
long-form guides. A useful comment usually names a complex in its first sentence
and drops the actual fact (the parking amount, the towing company, the leak) a
couple sentences later, so I needed chunks big enough to keep the name and the
detail together, sentence-level chunking would strand "parking is $130/mo" with
no apartment attached. I use `RecursiveCharacterTextSplitter`, which
breaks on blank-line → line → sentence boundaries first instead of cutting
mid-sentence, with the length measured in **real all-MiniLM tokens**,
and I prepend each thread's title to its text so the apartment is
present in the chunk. The 40-token overlap is insurance for facts that land near
a boundary.

I originally specced **500–700 tokens** in planning.md. When I built it and ran
my chunk-inspection tests, that produced only **34 chunks** (below the ~50 floor)
and **85% of chunks were longer than all-MiniLM-L6-v2's 256-token limit**, so
their tails would have been truncated at embedding time and never searched. I
dropped to 240/40 to fit the model's real window, and added a post-split merge
step that glues a stranded `Comment by u/…` header back onto its comment body.

**Final chunk count:** 84 chunks

### Sample chunks

Five real chunks from the store, each labeled with its source document:

**1 - `07_the_wilde_predatory_towing.txt`**

> The front desk staff is the epitome of unprofessionalism—rude, dismissive, and quick to hang up if you challenge their incompetence… my vehicle, which was parked in my paid reserved spot, was towed by Unlimited Recovery—a towing company contracted by The Wilde… Despite my vehicle being legally parked, it was removed and damaged while in their custody.

**2 - `03_hidden_fees_parking_costs.txt`**

> Comment by u/ComfortableOlive2003 (Apr 9, 2023): I lived at College Inn for a year… I lived in the 4x4 standard and it was $710 a month plus $50 a month for parking last year… My bf lives in a 4x4 at the standard and pays $900 a month plus $130 for parking plus $50 in fees plus more for utilities.

**3 - `09_the_standard_window_safety_hazards.txt`**

> No openable windows at The Standard. I lived at The Standard for 2 years and one of my biggest peeves was that we couldn't open any of the windows. Is this not a huge safety hazard? … there was a fire extinguisher set off right outside our door. There was a thick blanket of "smoke"… our only options would be to push into the smoke or smash out a window.

**4 - `02_avent_ferry_safety_prowler.txt`**

> [Safety Alert] Prowler / Casing Houses near Avent Ferry & Socket Dr… This individual has been repeatedly prowling and casing our house… He has knocked aggressively and tried to force entry. Raleigh Police have been notified.

**5 - `10_valentine_commons_infrastructure.txt`**

> Comment by u/RaphyTheFrenchDude (Sep 23, 2025): This is still fairly up to date, so far there's been issues with wifi, elevators not working, and plumbing issues, all were resolved within a week or so… rooms are very small and don't have overhead lighting so bring your own lamps.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers`, stored in a local
**ChromaDB** collection using cosine distance. I embed with normalized vectors
and keep `source`, `title`, `url`, and `chunk_index` as metadata on every chunk.
It runs locally with no API key or rate limits, which keeps the whole retrieval
side free, and it's plenty for short opinion text. I retrieve **top-k = 5**.

**Production tradeoff reflection:** If this were a real deployment and cost
wasn't a constraint, the thing I'd reconsider first is **accuracy on
domain-specific text**. all-MiniLM is small and general, and a lot of my
complaints use near-identical wording across different complexes ("management
never responds," "parking is a nightmare"), which is exactly where a bigger
model would do better at telling two apartments apart. Context length matters less here since
my chunks are short, and multilingual support isn't needed (all English). So the
real tradeoff is **latency + cost vs. retrieval precision**: local MiniLM is
instant and free but blunt; an API model is sharper but adds per-query cost and a
network round-trip.

---

## Grounded Generation

Generation lives in `query.py` and uses Groq's `llama-3.3-70b-versatile` at
`temperature=0`. Grounding is enforced two ways:

**System prompt grounding instruction.** The model is told (verbatim):

> Use ONLY information in the provided context. Do not use any outside or general
> knowledge. If the context does not contain enough information to answer the
> question, reply with EXACTLY this and nothing else: "I don't have enough
> information on that." Do not invent apartment names, prices, or facts that are
> not in the context.

Retrieved chunks are passed in a numbered, source-labeled context block
(`[1] (source: 07_the_wilde…txt)`), and the question is appended after it.

**Structural guards (not left to the model):**

- **Source attribution is programmatic.** I don't trust the LLM to cite, I build
  the source list from the metadata of the chunks that were actually retrieved
  (de-duplicated, in rank order). On a refusal the source list is emptied, so we
  never cite anything on an "I don't know."
- **Relevance gate.** If even the best chunk is farther than a cosine distance of
  0.85, I skip the LLM call entirely and refuse, so a totally out-of-domain
  question can't get a hallucinated answer.

**How source attribution is surfaced in the response:** the answer is returned with a `sources` list of
`{source, url}`, which the Gradio UI prints under the answer as bullet points.

---

## Query Interface

I built a **Gradio** web UI (`app.py`, run `python app.py` → http://localhost:7860).

- **Input:** a single text box labeled "Your question," plus an **Ask** button
  (pressing Enter submits too). There are clickable example questions underneath
  so you can try it without thinking one up.
- **Output:** an **Answer** box, a **Sources (retrieved threads)** box that lists
  the source files and their URLs, and a collapsible **Retrieved chunks** panel
  that shows the top chunks with their distance scores so you can actually see
  what the answer was grounded in.

**Sample interaction transcript (one full query):**

```
Your question:  Is there a safety concern near Avent Ferry and Socket Dr?

Answer:         Yes, there is a safety concern reported near Avent Ferry and
                Socket Dr, with a prowler casing houses and attempting to force
                entry, as reported by a student who shared doorbell camera
                footage. The student warned others to lock their doors and
                windows at all times.

Sources:        • 02_avent_ferry_safety_prowler.txt
                  (https://www.reddit.com/r/NCSU/comments/1trnl1b/…)
```

---

## Retrieval Test Results

Top chunks (source + cosine distance) for three of my eval queries, straight from
`python index.py`:

**Query: "Is there a safety concern reported near Avent Ferry and Socket Dr?"**

```
1. dist=0.470  02_avent_ferry_safety_prowler.txt   <-- expected
2. dist=0.633  05_spotting_fake_corporate_reviews.txt
3. dist=0.654  05_spotting_fake_corporate_reviews.txt
```

_Why these are relevant:_ the #1 chunk is the actual prowler-alert thread, and its
distance (0.470) sits well below everything after it (0.63+), so the right
document is clearly separated from the rest. The query and the chunk barely share
any exact words ("safety concern" vs. "prowling and casing"), semantic search
still finds it, which is the whole point of embeddings.

**Query: "What's the window or fire-safety problem at The Standard?"**

```
1. dist=0.526  09_the_standard_window_safety_hazards.txt   <-- expected
2. dist=0.586  09_the_standard_window_safety_hazards.txt   <-- expected
3. dist=0.724  04_hillsborough_commuting_subleases.txt
```

_Why these are relevant:_ the top two are both from the dedicated "No openable
windows at The Standard" thread and contain the exact fire/smoke/no-exit
complaint, and there's a clean gap (0.59 → 0.72) before anything off-topic shows
up. This is the system at its best, right topic, right source, on top.

**Query: "At Valentine Commons, what infrastructure problems do reviewers report?"**

```
1. dist=0.564  05_spotting_fake_corporate_reviews.txt
2. dist=0.585  07_the_wilde_predatory_towing.txt
3. dist=0.588  01_ncsu_general_recommendations.txt
4. dist=0.625  10_valentine_commons_infrastructure.txt   <-- expected
```

This one is weaker, the dedicated Valentine Commons thread only lands at #4.

A note on the scores: the Milestone 4 checkpoint suggests top results under 0.5.
Mine run **0.385–0.564**, so only two of five top hits are strictly under 0.5.
That's expected for all-MiniLM on question→review text (the question rarely shares
exact words with the review), and the retrieved chunks are still on-topic, so I
treated ~0.5 as a normal floor for this model rather than a bug to chase.

---

## Example Responses

**1 - in scope, sources visible:**

> **Q:** What's the window or fire-safety problem at The Standard?
> **A:** At The Standard, students raise the issue that there are no openable
> windows, which they consider a huge safety hazard, particularly in the event of
> a fire, as it would leave them with no other exit options.
> **Sources:** 09_the_standard_window_safety_hazards.txt

**2 - in scope, sources visible:**

> **Q:** Which management company do students say tows cars or damages vehicles?
> **A:** Students mention that the towing company contracted by **The Wilde**
> (Unlimited Recovery, now Integrity Towing) towed a car parked in a paid reserved
> spot and damaged it. **VIE** is also mentioned for towing cars parked legally.
> **Sources:** 07_the_wilde_predatory_towing.txt, 05_spotting_fake_corporate_reviews.txt

**3 - out of scope, refusal with no sources:**

> **Q:** What is the capital of France?
> **A:** I don't have enough information on that.
> **Sources:** (none)

---

## Evaluation Report

Run with `python evaluate.py`. Verdicts: **3 accurate, 2 partially accurate.**

| #   | Question                                         | Expected answer                                                                                                                               | System response (summarized)                                                                                                                               | Retrieval          | Accuracy               |
| --- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------- |
| 1   | Hidden fees / parking costs at budget complexes? | Parking billed separately, ~$40/mo (College Inn) up to $130–150/mo (Standard, Hillsborough), plus fees; budget rents ~$700–800, pricier ~$900 | Gave accurate per-complex parking numbers (College Inn $40, Stanhope $90/120, VC $75, Hillsborough $150 + $35 fee, Logan&Chamberlin $125), all from doc 05 | Partially relevant | **Partially accurate** |
| 2   | Safety concern near Avent Ferry & Socket Dr?     | Prowler/casing alert, doorbell footage, police notified                                                                                       | Yes — prowler casing houses, attempted forced entry, doorbell footage, lock your doors/windows                                                             | Relevant           | **Accurate**           |
| 3   | Which management company tows/damages cars?      | The Wilde — towing contract, legally-parked car towed & damaged, hostile mgmt                                                                 | The Wilde (towing co. Unlimited Recovery/Integrity) towed & damaged a reserved-spot car; also flagged VIE                                                  | Relevant           | **Accurate**           |
| 4   | Window / fire-safety problem at The Standard?    | Windows don't open → fire/smoke hazard, no secondary exit                                                                                     | Windows don't open, huge safety hazard in a fire, no other exit options                                                                                    | Relevant           | **Accurate**           |
| 5   | Infrastructure problems at Valentine Commons?    | Wi-Fi outages, elevators not working, plumbing, no overhead lighting                                                                          | Stairwells stank, small kitchens, gross elevators, no overhead lighting, noisy neighbors, iffy maintenance                                                 | Partially relevant | **Partially accurate** |

Notes:

- **Q1** answers the "parking costs" half very well but leans entirely on doc 05's
  price list; the $710/$130 figures in doc 03 never made the top 5, and it's
  light on the "advertised rent isn't the real cost" framing.
- **Q3** is correct on the main ask (The Wilde) but the parenthetical slightly
  conflates the _towing company's_ rename with the complex, traceable to a
  badly-punctuated sentence in the source thread.

---

## Failure Case Analysis

**Question that failed:** Q5 "At Valentine Commons, what infrastructure
problems do reviewers report?"

**What the system returned:** A grounded, on-topic-_sounding_ answer about
Valentine Commons, stank stairwells, small kitchens, "gross" elevators, no
overhead lighting, noisy neighbors. But it missed the specific infrastructure
list the question is really after (Wi-Fi outages, elevators _not working_,
plumbing issues) and conflated "elevators could be gross" (a cleanliness comment)
with "elevators not working" (an actual outage).

**Root cause:** The dedicated Valentine Commons
thread (`10_…`), which contains the exact chunk _"there's been issues with wifi,
elevators not working, and plumbing issues"_, only ranked **#4 (distance
0.625)**. The top three chunks came from **other** threads (`05`, `07`, and
especially `01`, where a user reminisces about living at VC years ago and
describes stank stairwells / small kitchens). Because Valentine Commons is
discussed across several threads, semantic similarity spread the score around,
and the older, longer, more prose-y `01` comment embedded _closer_ to my query
than the terse, list-style `10` chunk that actually has the facts. So generation
faithfully grounded itself on the wrong-but-on-topic chunks and produced a
plausible answer that's missing the real specifics, this is the
"cross-complex / cross-thread scattering" risk I called out in planning.md
happening for real.

**What you would change to fix it:** (1) raise `top-k` to ~8 so the dedicated thread's
chunk enters the context (I confirmed `10_…` is retrievable, just below the
cutoff); (2) add per-apartment **metadata filtering** so a question naming
"Valentine Commons" can prefer chunks whose source is the VC thread; or (3) try a
stronger embedding model that separates near-duplicate complaint wording better.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking
Strategy section in planning.md _before_ coding gave me concrete numbers and a
clear rationale to test the implementation against. Because I'd committed to
"chunk size in tokens, keep the apartment name bound to the detail, ~50+ chunks,"
I noticed immediately that my first build produced only 34 chunks and that most
exceeded the embedding model's window, the spec turned a silent quality problem
into an obvious, measurable miss. Without those written targets I probably would
have embedded oversized chunks and never known half of each one was being
truncated.

**One way your implementation diverged from the spec, and why:** I planned
**500–700 token** chunks with **100–150** overlap, and ended up at **240 / 40**.
The divergence was forced by a fact I didn't know when I wrote the spec:
all-MiniLM-L6-v2 only encodes the first 256 tokens, so my planned chunks would
have been silently cut in half at embedding time. I shrank the chunks to fit the
model's real window, scaled the overlap to keep the same ~15% ratio, and updated
planning.md to record why.

---

## AI Usage

**Instance 1**

- _What I gave the AI:_ my Documents and Chunking Strategy sections from
  planning.md, a sample of one raw thread so it could see the Reddit boilerplate,
  and my 500–700 token / 100–150 overlap target.
- _What it produced:_ `pipeline.py`, a loader, a cleaning pass that strips the
  separators / vote counts / comment banners and fixes HTML escapes, and a
  `RecursiveCharacterTextSplitter` measured in real MiniLM tokens, plus a test
  file that inspects the chunks.
- _What I changed or overrode:_ the tests showed my own spec was wrong, 34
  chunks and 85% over the 256-token limit, so I overrode the chunk size down to
  240/40 to fit the embedding window and added a merge step for stranded
  `Comment by…` headers. I also tightened the comment-header regex after it left
  `[deleted]` users un-cleaned.

**Instance 2**

- _What I gave the AI:_ my grounding requirement from planning.md (answer only
  from retrieved context, otherwise refuse), the output format I wanted (answer +
  source list), and the Gradio skeleton from the instructions, and asked it to
  wire retrieval into Groq's `llama-3.3-70b-versatile`.
- _What it produced:_ a first version that put the source list together by asking
  the model to cite its own sources in the answer text, and relied entirely on
  the system prompt to keep it grounded.
- _What I changed or overrode:_ I didn't want citations depending on the model
  behaving, so I made source attribution **programmatic**, the sources come from
  the metadata of the chunks that were actually retrieved,
  and they're emptied on a refusal. I also added a fixed refusal sentinel
  the prompt must return verbatim, plus a distance-based relevance gate that
  refuses without even calling the LLM when nothing retrieved is close enough.

---

## Demo Video

A 3–5 minute walkthrough covering three queries with their source citations: one
where retrieval and generation both work well (The Standard), one that shows the
refusal being enforced in the prompt (the out-of-scope "capital of France"), and
one where the system struggles (Valentine Commons). It finishes with a walk
through the evaluation report.

**[Watch the demo (demo.mp4)](demo.mp4)**

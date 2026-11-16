"""Synthetic personal corpus + labeled queries.

I don't want my real inbox in a public repo, so this builds a fake one for a
made-up person ("Alex Rivera") with the same *shape* as mine: lecture notes,
project meeting notes, design docs, a lease, receipts, flight emails, etc.

Things I did on purpose so the benchmark isn't trivially easy:
  * lots of near-duplicates (several flights, several orders, several projects
    with similar meeting emails) so the retriever has to use the details
  * most queries are paraphrased ("when does my flight to Vancouver leave")
    instead of copying words from the doc ("YYC -> YVR departs 07:45")
  * a few queries are exact-identifier lookups (order numbers, booking refs)
    where keyword search *should* win, so BM25 isn't set up to lose

Everything is seeded, so `python scripts/generate_corpus.py` always produces
the same files that are committed under data/.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path

from agentic_search.types import Document

OWNER = "Alex Rivera"
OWNER_EMAIL = "alex@example.com"

PEOPLE = [
    "Priya Natarajan", "Marcus Oyelaran", "Elena Kowalczyk", "Daniel Mendes",
    "Aiko Tanaka", "Sam Whitfield", "Fatima Rahimi", "Owen Gallagher",
    "Lucia Ferreira", "Noah Brandt", "Grace Liang", "Tomas Novak",
]

# codename -> one-line description, used for meeting notes, design docs and emails
PROJECTS = {
    "Heron": "a campus ride-share matching service",
    "Kestrel": "a CLI that syncs lecture slides into Markdown notes",
    "Lantern": "a budgeting dashboard that reads bank CSV exports",
    "Quarry": "a web scraper that tracks internship postings",
    "Bramble": "a plant-watering reminder app with a Raspberry Pi sensor",
}

# (city, airport code, a word people actually say instead of the code)
DESTINATIONS = [
    ("Vancouver", "YVR", "the coast"),
    ("Toronto", "YYZ", "Ontario"),
    ("Montreal", "YUL", "Quebec"),
    ("Seattle", "SEA", "Washington"),
    ("Halifax", "YHZ", "Nova Scotia"),
    ("Denver", "DEN", "Colorado"),
]

PRODUCTS = [
    ("mechanical keyboard", "Keychron K8 mechanical keyboard (brown switches)"),
    ("monitor", "27-inch 1440p IPS monitor"),
    ("running shoes", "trail running shoes, size 10"),
    ("desk lamp", "LED desk lamp with USB-C charging port"),
    ("headphones", "noise-cancelling over-ear headphones"),
    ("textbook", "Operating System Concepts, 10th edition (used)"),
    ("backpack", "28L commuter backpack with laptop sleeve"),
    ("water bottle", "insulated steel water bottle, 1L"),
]

# The lecture-note concept bank. Each entry: (course, topic, note text, question
# I'd actually type later when studying). These are real CS facts so the notes
# read like notes, but the wording in the question deliberately avoids the
# note's key terms where possible.
CONCEPTS = [
    ("CPSC 457", "Round robin scheduling",
     "Round robin gives each ready process a fixed time quantum, then preempts it and moves it to the back of the ready queue. A quantum that is too small means most CPU time goes to context switches; too large and it degrades into FCFS. Rule of thumb from lecture: about 80% of CPU bursts should finish within one quantum.",
     "how big should the time slice be when each process takes turns on the cpu"),
    ("CPSC 457", "Deadlock conditions",
     "Deadlock needs all four Coffman conditions at once: mutual exclusion, hold and wait, no preemption, and circular wait. Breaking any one of them prevents deadlock. Lock ordering is the practical fix because it removes circular wait.",
     "what four things have to be true for processes to get permanently stuck waiting on each other"),
    ("CPSC 457", "Paging and TLB",
     "Paging splits virtual memory into fixed-size pages mapped to physical frames through a page table. The TLB caches recent translations; effective access time = hit_ratio * (tlb + mem) + (1 - hit_ratio) * (tlb + 2*mem). A 98% hit ratio keeps the overhead small.",
     "formula for effective memory access time with the translation cache"),
    ("CPSC 457", "Semaphores vs mutexes",
     "A mutex has an owner and only the locking thread may unlock it. A counting semaphore has no owner and tracks a number of available resources, so it can be used for signalling between threads. The bounded buffer solution uses two counting semaphores (empty, full) plus one mutex.",
     "difference between a lock that has an owner and a counter used for signalling"),
    ("CPSC 457", "Page replacement",
     "LRU replaces the page that has not been used for the longest time and approximates OPT well, but exact LRU is expensive, so the clock (second chance) algorithm uses a reference bit instead. FIFO can show Belady's anomaly: more frames, more faults.",
     "why can adding more memory frames sometimes cause more page faults"),
    ("CPSC 457", "Fork and exec",
     "fork() duplicates the calling process; the child gets return value 0 and the parent gets the child's PID. exec() replaces the current process image with a new program. A shell runs fork, then exec in the child, then waitpid in the parent.",
     "how does a shell start a new program and wait for it"),
    ("CPSC 413", "Master theorem",
     "For T(n) = aT(n/b) + f(n): compare f(n) with n^(log_b a). If f is polynomially smaller, T = Theta(n^(log_b a)); if equal up to log factors, multiply by log n; if bigger and regular, T = Theta(f(n)). Merge sort is case 2: Theta(n log n).",
     "shortcut for solving divide and conquer recurrences"),
    ("CPSC 413", "Dijkstra's algorithm",
     "Dijkstra finds single-source shortest paths when every edge weight is non-negative. With a binary heap it runs in O((V + E) log V). Negative edges break the greedy invariant, so use Bellman-Ford there.",
     "which shortest path algorithm to use when some edges are negative"),
    ("CPSC 413", "Dynamic programming: knapsack",
     "0/1 knapsack: dp[i][w] = max(dp[i-1][w], v_i + dp[i-1][w - w_i]). It is O(nW), which is pseudo-polynomial because W is a number, not an input length. Fractional knapsack is solvable greedily by value per weight.",
     "why is the bag packing dp not actually polynomial time"),
    ("CPSC 413", "NP-completeness reductions",
     "To show problem X is NP-complete: show X is in NP (verify a certificate in polynomial time), then reduce a known NP-complete problem to X in polynomial time. Direction matters: reduce FROM the known hard problem TO X.",
     "which direction do I reduce when proving a problem is hard"),
    ("CPSC 413", "Minimum spanning trees",
     "Kruskal sorts edges and adds the cheapest edge that doesn't form a cycle, using union-find; O(E log E). Prim grows one tree from a start vertex with a priority queue. Both rely on the cut property.",
     "algorithm that sorts edges and uses union find to connect all vertices cheaply"),
    ("CPSC 471", "Normalization: 3NF vs BCNF",
     "BCNF requires every non-trivial functional dependency X -> Y to have X as a superkey. 3NF relaxes this by also allowing Y to be part of a candidate key. Decomposing to BCNF is always lossless but may not preserve dependencies; 3NF synthesis preserves both.",
     "when should a schema be decomposed to the stricter normal form instead of the third one"),
    ("CPSC 471", "B+ tree indexes",
     "B+ trees keep all records in the leaves and link leaves together, so range scans are a single walk across the leaf level. Fanout is high (hundreds of keys per node) so the tree is usually only 3-4 levels deep even for millions of rows.",
     "why are range queries fast on the standard database index"),
    ("CPSC 471", "Transactions and isolation",
     "ACID: atomicity, consistency, isolation, durability. Read committed prevents dirty reads but allows non-repeatable reads; repeatable read stops those but can still allow phantoms; serializable prevents all three. Postgres implements serializable with SSI.",
     "which isolation level still lets new rows appear between two identical reads"),
    ("CPSC 471", "Join algorithms",
     "Nested loop join is fine when the inner side has an index. Hash join builds a hash table on the smaller input and probes with the larger, good for equality joins. Sort-merge join wins when both inputs are already sorted on the join key.",
     "best way to combine two big tables on an equality condition with no index"),
    ("CPSC 441", "TCP congestion control",
     "TCP Reno starts with slow start (cwnd doubles each RTT) until ssthresh, then switches to additive increase. On triple duplicate ACKs it halves cwnd (fast recovery); on timeout cwnd drops to 1 MSS.",
     "what happens to the sending window after three duplicate acks"),
    ("CPSC 441", "DNS resolution",
     "A stub resolver asks a recursive resolver, which walks root -> TLD -> authoritative servers and caches each answer for its TTL. Most lookups are answered from cache, which is why changing a record can take hours to propagate.",
     "why do domain name changes take a while to show up everywhere"),
    ("CPSC 441", "Subnetting",
     "A /26 has 64 addresses, 62 usable hosts after removing network and broadcast. CIDR notation /n means the first n bits are the network prefix. 192.168.1.64/26 covers .64 to .127.",
     "how many usable hosts in a slash 26 network"),
    ("CPSC 441", "HTTP/2 vs HTTP/1.1",
     "HTTP/2 multiplexes many streams over one TCP connection and compresses headers with HPACK, which removes HTTP/1.1's application-level head-of-line blocking. TCP-level head-of-line blocking is still there, which is what QUIC/HTTP/3 fixes.",
     "what problem does the newest web protocol over udp fix that version 2 still has"),
    ("CPSC 501", "Bias-variance tradeoff",
     "High bias models underfit (training and validation error both high); high variance models overfit (low training error, high validation error). Regularization, more data and simpler models reduce variance.",
     "my training error is low but validation error is high, what does that mean"),
    ("CPSC 501", "Cross-entropy loss",
     "For classification we minimize cross-entropy, -sum y log p. Combined with softmax its gradient with respect to the logits is just p - y, which is why it trains better than squared error on probabilities.",
     "why is the gradient so simple when you pair softmax with log loss"),
    ("CPSC 501", "Attention",
     "Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V. Dividing by sqrt(d_k) keeps the dot products from growing with dimension and saturating the softmax. Multi-head attention runs several of these in parallel on projected subspaces.",
     "why divide by the square root of the key dimension in transformers"),
    ("CPSC 501", "Precision and recall",
     "Precision = TP / (TP + FP), recall = TP / (TP + FN). For a search system recall@k asks whether the relevant item shows up in the top k at all; MRR rewards putting it near the top.",
     "metric for whether the right result appears anywhere in the first few results"),
    ("CPSC 501", "Embeddings and cosine similarity",
     "Sentence embeddings map text to dense vectors so that similar meanings land close together. If vectors are L2-normalized, cosine similarity equals the inner product, so an inner-product index gives cosine search for free.",
     "if vectors are unit length how does dot product relate to the angle between them"),
    ("CPSC 501", "Gradient descent learning rate",
     "Too high a learning rate makes the loss oscillate or diverge; too low and training crawls. Warmup followed by cosine decay is a common schedule for transformers.",
     "loss keeps bouncing around and blowing up during training, what knob to change"),
]

RECIPES = [
    ("banana bread", "350F (175C) for 60 minutes", "3 very ripe bananas, 1/3 cup melted butter, 3/4 cup sugar, 1 egg, 1 tsp baking soda, 1.5 cups flour",
     "how long does the banana loaf bake and at what heat"),
    ("chickpea curry", "simmer 20 minutes", "2 cans chickpeas, 1 can coconut milk, onion, garlic, ginger, 2 tbsp curry paste, spinach at the end",
     "what goes in the coconut milk dish with garbanzo beans"),
    ("overnight oats", "fridge overnight, no cooking", "1/2 cup oats, 1/2 cup milk, 1/4 cup greek yogurt, 1 tbsp chia, honey",
     "breakfast I prep the night before with chia"),
    ("shakshuka", "simmer sauce 10 min, eggs covered 6-8 min", "crushed tomatoes, red pepper, onion, cumin, paprika, 4 eggs, feta",
     "eggs poached in tomato sauce recipe"),
    ("pizza dough", "rise 2 hours, bake 500F for 10 minutes", "500g flour, 325g water, 10g salt, 3g instant yeast, olive oil",
     "hydration and yeast amounts for homemade pizza base"),
]

BOOKS = [
    ("Designing Data-Intensive Applications", "Martin Kleppmann",
     "Big takeaway: replication lag is the root of most 'read your own writes' bugs. Chapter 7 on transactions finally made write skew make sense to me."),
    ("The Pragmatic Programmer", "Hunt and Thomas",
     "Tracer bullets: build a thin end-to-end slice first, then fill in. That's basically how I want to build every side project from now on."),
    ("Atomic Habits", "James Clear",
     "Habit stacking: attach a new habit to an existing one. I'm trying 'after I make coffee, I do 10 minutes of Anki'."),
    ("Thinking, Fast and Slow", "Daniel Kahneman",
     "System 1 vs system 2. The planning fallacy section explains why every one of my assignment estimates is wrong by 2x."),
]


def _date(rng: random.Random, start: date, span_days: int) -> date:
    return start + timedelta(days=rng.randrange(span_days))


class _Builder:
    """Accumulates documents and their eval queries while keeping ids stable."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.docs: list[Document] = []
        self.queries: list[dict] = []
        self._counter: dict[str, int] = {}

    def add(self, source: str, title: str, d: date, body: str, meta: dict | None = None) -> str:
        n = self._counter.get(source, 0) + 1
        self._counter[source] = n
        doc_id = f"{source}-{n:04d}"
        self.docs.append(Document(doc_id, source, title, d.isoformat(), body.strip(), meta or {}))
        return doc_id

    def ask(self, query: str, relevant: list[str], kind: str) -> None:
        qid = f"q{len(self.queries) + 1:03d}"
        self.queries.append({"qid": qid, "query": query, "relevant": relevant, "kind": kind})


# ---------------------------------------------------------------- notes

def _lecture_notes(b: _Builder) -> None:
    term_start = date(2026, 1, 8)
    for course, topic, text, question in CONCEPTS:
        d = _date(b.rng, term_start, 100)
        # little bits of real-notes noise so the note isn't only the fact
        extra = b.rng.choice([
            "Prof said this will definitely be on the midterm.",
            "Need to redo the practice problem from the slides.",
            "Ask in office hours: still a bit fuzzy on the edge cases.",
            "Tutorial worked through two examples of this.",
            "Compare with the textbook version, notation is different.",
        ])
        body = f"{course} lecture notes - {topic}\n\n{text}\n\n{extra}"
        doc_id = b.add("note", f"{course}: {topic}", d, body, {"course": course})
        b.ask(question, [doc_id], "concept")


def _recipe_notes(b: _Builder) -> None:
    for dish, method, ingredients, question in RECIPES:
        d = _date(b.rng, date(2025, 6, 1), 400)
        body = f"Recipe: {dish}\nIngredients: {ingredients}\nMethod: {method}.\nNotes: halve the sugar next time / double the batch for meal prep."
        doc_id = b.add("note", f"Recipe - {dish}", d, body)
        b.ask(question, [doc_id], "paraphrase")


def _book_notes(b: _Builder) -> None:
    for title, author, take in BOOKS:
        d = _date(b.rng, date(2025, 3, 1), 500)
        doc_id = b.add("note", f"Book notes: {title}", d, f"{title} by {author}\n\n{take}")
    # one question per book, but phrased about the idea, not the title
    ids = [doc.doc_id for doc in b.docs if doc.title.startswith("Book notes")]
    b.ask("which book explained why my time estimates are always too optimistic", [ids[3]], "paraphrase")
    b.ask("book with the idea of building a thin end to end slice first", [ids[1]], "paraphrase")
    b.ask("notes on attaching a new habit to something I already do", [ids[2]], "paraphrase")


def _gym_log(b: _Builder) -> None:
    # repetitive, low-value notes: realistic noise and good distractors
    lifts = ["squat", "bench press", "deadlift", "overhead press", "barbell row"]
    start = date(2026, 1, 5)
    for week in range(20):
        d = start + timedelta(days=7 * week + b.rng.randrange(3))
        lift = lifts[week % len(lifts)]
        weight = 60 + 2.5 * week + b.rng.choice([0, 2.5, 5])
        b.add("note", f"Gym log {d.isoformat()}", d,
              f"Workout {d.isoformat()}\nMain lift: {lift} 5x5 @ {weight}kg\nAccessories: pull-ups, planks.\nFelt {b.rng.choice(['strong', 'tired', 'ok', 'sore from last time'])}.")


def _project_meeting_notes(b: _Builder) -> None:
    decisions = {
        "Heron": ("use Postgres with PostGIS instead of DynamoDB", "we need radius queries for pickup points and the free tier of DynamoDB doesn't do geo"),
        "Kestrel": ("parse PDFs with pdfplumber instead of PyPDF2", "PyPDF2 kept merging the two-column slide layouts into one line"),
        "Lantern": ("store transactions in SQLite locally instead of a hosted DB", "bank data should never leave the laptop"),
        "Quarry": ("poll each job board at most once every 6 hours", "one site rate-limited us and returned 429s for a whole day"),
        "Bramble": ("switch the soil sensor from resistive to capacitive", "the resistive probe corroded after three weeks"),
    }
    questions = {
        "Heron": "why didn't we go with the amazon nosql database for the ride share app",
        "Kestrel": "what library did we pick for reading the slide pdfs and why",
        "Lantern": "where does the budgeting tool keep the bank data",
        "Quarry": "how often is the internship scraper allowed to hit each site",
        "Bramble": "what happened to the first moisture probe on the plant project",
    }
    for name, (decision, why) in decisions.items():
        attendees = b.rng.sample(PEOPLE, 3)
        d = _date(b.rng, date(2026, 2, 1), 150)
        body = (f"Project {name} sync - {d.isoformat()}\nAttendees: {', '.join(attendees)}, {OWNER}\n\n"
                f"Decision: {decision}. Reason: {why}.\n"
                f"Action items: {attendees[0]} writes the migration plan, {OWNER} updates the README.")
        doc_id = b.add("note", f"{name} meeting notes", d, body, {"project": name})
        b.ask(questions[name], [doc_id], "paraphrase")
        # a second, boring sync for each project = hard negative for the query above
        d2 = d + timedelta(days=14)
        b.add("note", f"{name} meeting notes", d2,
              f"Project {name} sync - {d2.isoformat()}\nAttendees: {', '.join(b.rng.sample(PEOPLE, 2))}, {OWNER}\n\n"
              f"Status update only. Demo went fine, no new decisions. Next sync in two weeks.")


# ---------------------------------------------------------------- docs

def _design_docs(b: _Builder) -> None:
    owners = {}
    for name, desc in PROJECTS.items():
        owner = b.rng.choice(PEOPLE)
        owners[name] = owner
        d = _date(b.rng, date(2026, 1, 15), 60)
        body = (f"{name} design doc\n\nOverview: {name} is {desc}.\nOwner: {owner}\n\n"
                f"Goals: ship an MVP by end of term, keep hosting under $10/month.\n"
                f"Non-goals: mobile app, multi-tenant accounts.\n"
                f"Open questions: auth provider, how to back up data.")
        doc_id = b.add("doc", f"{name} design doc", d, body, {"project": name})
    b.ask("who owns the plant watering project", [f"doc-{list(PROJECTS).index('Bramble') + 1:04d}"], "paraphrase")
    b.ask("design doc for the tool that turns lecture slides into markdown", [f"doc-{list(PROJECTS).index('Kestrel') + 1:04d}"], "paraphrase")


def _lease_and_admin_docs(b: _Builder) -> None:
    lease = b.add("doc", "Apartment lease 2025-2026", date(2025, 8, 20), """
Residential tenancy agreement - Unit 1204, 88 Brentwood Rd NW
Tenant: Alex Rivera. Landlord: Northgate Property Management.
Term: September 1 2025 to August 31 2026. Monthly rent $1,450 due on the 1st.
Security deposit: $1,450, returned within 10 days of move-out minus damages.
Pets: one cat or small dog allowed with a $300 non-refundable pet fee.
Notice: tenant must give 60 days written notice before the end of the term.
Utilities: water and heat included, electricity and internet paid by tenant.""")
    b.ask("how much warning do I need to give before moving out of my apartment", [lease], "paraphrase")
    b.ask("am I allowed to have a cat in my place", [lease], "paraphrase")

    warranty = b.add("doc", "Laptop warranty certificate", date(2025, 9, 2), """
Extended warranty - 14-inch laptop, serial ending 7Q2F
Coverage: 3 years parts and labour from purchase date (Sept 2 2025).
Accidental damage: one claim per year, $99 deductible.
Battery covered only if capacity drops below 80% within the first year.""")
    b.ask("is a cracked laptop screen covered and what does it cost me", [warranty], "paraphrase")

    tax = b.add("doc", "Tax checklist 2025", date(2026, 2, 10), """
Things to collect before filing:
- T4 from summer internship
- T2202 tuition slip from the university (download from student centre)
- RRSP contribution receipts if any
- Rent receipts are NOT needed in Alberta
Deadline: April 30. Use the free NETFILE-certified software.""")
    b.ask("what forms do I need before doing my taxes", [tax], "paraphrase")

    resume_tips = b.add("doc", "Resume feedback from career centre", date(2026, 9, 15), """
Feedback from the co-op advisor:
- lead each bullet with an action verb and a number where possible
- move projects above coursework
- keep it to one page; cut the high school awards
- link GitHub and make sure the top repos have READMEs""")
    b.ask("what did the advisor say about my cv", [resume_tips], "paraphrase")


# ---------------------------------------------------------------- emails

def _email(b: _Builder, sender: str, subject: str, d: date, body: str) -> str:
    meta = {"from": sender, "to": OWNER_EMAIL, "subject": subject}
    full = f"From: {sender}\nTo: {OWNER_EMAIL}\nSubject: {subject}\n\n{body}"
    return b.add("email", subject, d, full, meta)


def _flight_emails(b: _Builder) -> None:
    for city, code, alias in DESTINATIONS:
        d = _date(b.rng, date(2026, 1, 1), 300)
        dep = f"{b.rng.randrange(6, 21):02d}:{b.rng.choice(['05', '20', '35', '50'])}"
        ref = "".join(b.rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
        travel = d + timedelta(days=b.rng.randrange(10, 40))
        body = (f"Booking reference: {ref}\n\nItinerary\n  NW{b.rng.randrange(100, 999)}  YYC -> {code}  "
                f"{travel.isoformat()}  departs {dep}\nFare: Basic. 1 personal item included, checked bags extra.\n"
                f"Check-in opens 24 hours before departure.")
        doc_id = _email(b, "Northwind Air <no-reply@northwind-air.example>",
                        f"Your trip confirmation {ref}", d, body)
        # the paraphrased question uses the city name, the email only has the code
        b.ask(f"what time does my flight to {city} take off", [doc_id], "paraphrase")
        if city in ("Halifax", "Denver"):
            b.ask(f"booking {ref}", [doc_id], "exact")


def _order_emails(b: _Builder) -> None:
    for short, full in PRODUCTS:
        d = _date(b.rng, date(2026, 1, 1), 300)
        order = f"{b.rng.randrange(100000, 999999)}"
        eta = d + timedelta(days=b.rng.randrange(2, 8))
        carrier = b.rng.choice(["Canada Post", "Purolator", "UPS"])
        body = (f"Good news - order #{order} has shipped!\n\nItems: {full}\nCarrier: {carrier}\n"
                f"Estimated delivery: {eta.strftime('%A, %B %d')}\nTrack your package in your account.")
        doc_id = _email(b, "Shop Orders <orders@shop.example>", f"Order #{order} shipped", d, body)
        b.ask(f"when is the {short} I bought supposed to arrive", [doc_id], "paraphrase")
        if short in ("monitor", "backpack", "textbook"):
            b.ask(f"order {order}", [doc_id], "exact")


def _meeting_emails(b: _Builder) -> None:
    rooms = ["ICT 516", "MS 211", "TFDL 4th floor study room", "ENG A 103", "online (Zoom)"]
    for name in PROJECTS:
        sender = b.rng.choice(PEOPLE)
        d = _date(b.rng, date(2026, 3, 1), 200)
        room = b.rng.choice(rooms)
        day = b.rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"])
        hour = b.rng.choice(["10am", "1pm", "3pm", "4:30pm"])
        first = sender.split()[0]
        body = (f"Hi Alex,\n\nCan we move the {name} sync to {day} at {hour}? I booked {room}.\n"
                f"I'll bring the updated diagrams.\n\nThanks,\n{first}")
        doc_id = _email(b, f"{sender} <{first.lower()}@example.com>", f"{name} sync - new time", d, body)
        b.ask(f"where is the {name} meeting happening now", [doc_id], "paraphrase")


def _course_emails(b: _Builder) -> None:
    exams = [
        ("CPSC 457", "Operating Systems", "processes, threads, scheduling, synchronization and deadlock", "ST 140"),
        ("CPSC 413", "Design and Analysis of Algorithms", "recurrences, greedy algorithms and dynamic programming", "SA 104"),
        ("CPSC 471", "Database Management Systems", "ER modelling, relational algebra, SQL and normalization", "KNB 132"),
        ("CPSC 441", "Computer Networks", "the application and transport layers, including TCP", "ICT 102"),
    ]
    subject_words = {"CPSC 457": "OS", "CPSC 413": "algorithms", "CPSC 471": "databases", "CPSC 441": "networking"}
    for code, name, covers, room in exams:
        d = _date(b.rng, date(2026, 2, 1), 40)
        exam_day = d + timedelta(days=b.rng.randrange(7, 20))
        body = (f"Hello {code} students,\n\nReminder that the midterm for {name} is on "
                f"{exam_day.strftime('%B %d')} in {room}, 6:00-7:30pm.\nIt covers {covers}.\n"
                f"One double-sided handwritten cheat sheet is allowed. No calculators.\n\n- Course instructor")
        doc_id = _email(b, f"{code} Instructor <{code.replace(' ', '').lower()}@ucalgary.example>",
                        f"{code} midterm information", d, body)
        b.ask(f"what topics are on the {subject_words[code]} midterm", [doc_id], "paraphrase")


def _landlord_emails(b: _Builder) -> None:
    sink = _email(b, "Northgate Property Management <maint@northgate.example>",
                  "Maintenance visit - Unit 1204", date(2026, 4, 3),
                  "Hi Alex,\n\nA plumber from Bow River Plumbing will come Tuesday April 7 between 9am and noon "
                  "to repair the leaking kitchen faucet. Someone does not need to be home; we will use the master key.\n\nNorthgate")
    b.ask("when is someone coming to fix the sink", [sink], "paraphrase")
    _email(b, "Northgate Property Management <office@northgate.example>", "Building notice: elevator",
           date(2026, 5, 11), "The east elevator will be out of service May 14-15 for annual inspection. Please use the west elevator.")
    rent = _email(b, "Northgate Property Management <office@northgate.example>", "Lease renewal offer",
                  date(2026, 6, 20),
                  "Hi Alex,\n\nWe'd like to offer a 12-month renewal starting September 1 2026 at $1,525/month "
                  "(up from $1,450). Please let us know by July 31.\n\nNorthgate")
    b.ask("how much is my rent going up next year", [rent], "paraphrase")


def _recruiting_emails(b: _Builder) -> None:
    companies = [
        ("Lumen Robotics", "systems design", "45 minutes", "Grace Liang"),
        ("Fernwood Analytics", "SQL and a take-home review", "60 minutes", "Tomas Novak"),
        ("Pinecrest Games", "C++ and data structures", "50 minutes", "Owen Gallagher"),
    ]
    for company, topic, length, interviewer in companies:
        d = _date(b.rng, date(2026, 9, 1), 60)
        when = d + timedelta(days=b.rng.randrange(5, 15))
        body = (f"Hi Alex,\n\nThanks for applying to {company}! Your technical interview is scheduled for "
                f"{when.strftime('%A %B %d')} at 2:00pm MT ({length}) with {interviewer}.\n"
                f"Expect questions on {topic}. A video link will follow.\n\nBest,\n{company} Recruiting")
        doc_id = _email(b, f"{company} Recruiting <jobs@{company.split()[0].lower()}.example>",
                        f"Interview scheduled - {company}", d, body)
    ids = [doc.doc_id for doc in b.docs if doc.title.startswith("Interview scheduled")]
    b.ask("what should I study for the robotics company interview", [ids[0]], "paraphrase")
    b.ask("who is interviewing me at the game studio", [ids[2]], "paraphrase")
    b.ask("how long is the analytics interview", [ids[1]], "paraphrase")


def _subscription_emails(b: _Builder) -> None:
    subs = [
        ("CloudVault", "2TB storage plan", "$11.99/month", "online storage"),
        ("StreamBox", "Standard plan", "$16.49/month", "video streaming"),
        ("GymPlus", "student membership", "$34.00/month", "gym"),
    ]
    for brand, plan, price, what in subs:
        d = _date(b.rng, date(2026, 1, 1), 300)
        body = (f"Your {brand} {plan} will renew on {(d + timedelta(days=7)).strftime('%B %d')} for {price}.\n"
                f"Payment method: card ending 4242. Manage or cancel anytime in account settings.")
        doc_id = _email(b, f"{brand} Billing <billing@{brand.lower()}.example>", f"{brand} renewal reminder", d, body)
        b.ask(f"how much am I paying for {what}", [doc_id], "paraphrase")


def _newsletters(b: _Builder) -> None:
    # pure noise: promotional email that shares vocabulary with real queries
    # (flights, orders, deadlines) but is never the answer
    blurbs = [
        ("Northwind Air <deals@northwind-air.example>", "Fall seat sale: fly to Vancouver from $129",
         "Book by Friday for fares to Vancouver, Toronto and Seattle. Flights must be taken before December 15."),
        ("Shop Deals <deals@shop.example>", "Keyboards and monitors up to 30% off",
         "This week only: mechanical keyboards, monitors and headphones on sale. Free shipping over $50."),
        ("Campus Life <news@ucalgary.example>", "This week on campus",
         "Midterm study spaces are open late in TFDL. Free coffee Wednesday. Career fair next Thursday."),
        ("CloudVault <news@cloudvault.example>", "New: shared family folders",
         "You can now share folders with up to 5 people on any storage plan."),
        ("GymPlus <news@gymplus.example>", "New spin classes",
         "Spin classes now run Monday and Wednesday at 6pm. Members book for free."),
    ]
    for _ in range(4):  # several copies over the year, like real newsletters
        for sender, subject, body in blurbs:
            _email(b, sender, subject, _date(b.rng, date(2026, 1, 1), 300), body)


def _misc_notes(b: _Builder) -> None:
    wifi = b.add("note", "Home network setup", date(2025, 9, 3), """
Router admin page: 192.168.0.1 (admin password is in my password manager, not here)
Network name: rivera-5g. Guest network: rivera-guest, turned off by default.
Printer has a static IP of 192.168.0.40 so the laptop can always find it.""")
    b.ask("what address did I give the printer", [wifi], "paraphrase")

    ideas = b.add("note", "Side project ideas", date(2026, 7, 2), """
- search engine over my own notes + email (this project!)
- bus arrival predictor for route 9 using the city's open GTFS feed
- tool that turns a recipe url into a grocery list grouped by aisle""")
    b.ask("what was my idea involving transit data", [ideas], "paraphrase")

    car = b.add("note", "Car maintenance", date(2026, 5, 18), """
2015 Civic. Oil change at 142,300 km on May 18 2026 (synthetic 0W-20).
Next oil change around 150,000 km. Winter tires stored at the dealer, swap in mid October.""")
    b.ask("when do I need to get the oil changed again", [car], "paraphrase")


def build_corpus(seed: int = 457) -> tuple[list[Document], list[dict]]:
    b = _Builder(seed)
    # order matters for stable ids, so don't shuffle these
    _lecture_notes(b)
    _recipe_notes(b)
    _book_notes(b)
    _gym_log(b)
    _project_meeting_notes(b)
    _misc_notes(b)
    _design_docs(b)
    _lease_and_admin_docs(b)
    _flight_emails(b)
    _order_emails(b)
    _meeting_emails(b)
    _course_emails(b)
    _landlord_emails(b)
    _recruiting_emails(b)
    _subscription_emails(b)
    _newsletters(b)
    return b.docs, b.queries


def write_corpus(out_dir: Path, seed: int = 457) -> tuple[int, int]:
    docs, queries = build_corpus(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w") as f:
        for doc in docs:
            f.write(json.dumps(asdict(doc)) + "\n")
    with open(out_dir / "queries.jsonl", "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    return len(docs), len(queries)

"""
Milestone 6 — Evaluation harness.

Runs the 5 evaluation questions from planning.md through the full system
(retrieve -> ground -> generate) and prints, for each: the question, the
generated answer, whether it refused, the sources cited, and the top-5
retrieved chunks with their cosine distances. The output is what backs the
Evaluation Report in README.md.

    python evaluate.py
"""

import sys

from query import ask

# (question, expected-answer summary, expected source file(s)) — from planning.md
EVAL = [
    ("What are the hidden fees and parking costs at the budget complexes students mention?",
     "Rent isn't the real cost: parking is billed separately and varies by complex "
     "(~$40/mo College Inn up to $130-150/mo Standard/Hillsborough), plus utility/admin "
     "fees; budget 4x4 rents ~$700-800, pricier builds ~$900.",
     ["03_hidden_fees_parking_costs.txt", "05_spotting_fake_corporate_reviews.txt"]),

    ("Is there a safety concern reported near Avent Ferry and Socket Dr?",
     "Yes — a prowler/casing-houses alert with doorbell-camera footage; Raleigh Police "
     "were notified.",
     ["02_avent_ferry_safety_prowler.txt"]),

    ("Which management company do students say tows cars or damages vehicles?",
     "The Wilde — predatory towing contract, a legally parked car towed and damaged, "
     "hostile management.",
     ["07_the_wilde_predatory_towing.txt"]),

    ("What's the window or fire-safety problem students raise about The Standard?",
     "The windows don't open, which students flag as a fire/smoke hazard with no "
     "secondary exit in the newer build.",
     ["09_the_standard_window_safety_hazards.txt"]),

    ("At Valentine Commons, what infrastructure problems do reviewers report?",
     "Wi-Fi outages, elevators not working, plumbing issues, and no overhead room lighting.",
     ["10_valentine_commons_infrastructure.txt"]),
]


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    for i, (q, expected, exp_sources) in enumerate(EVAL, 1):
        r = ask(q)
        print("#" * 80)
        print(f"Q{i}: {q}")
        print(f"\nEXPECTED ({', '.join(exp_sources)}):\n  {expected}")
        print(f"\nSYSTEM ANSWER (refused={r['refused']}):\n  {r['answer']}")
        cited = ", ".join(s["source"] for s in r["sources"]) or "(none)"
        print(f"\nCITED SOURCES: {cited}")
        print("\nTOP-5 RETRIEVED:")
        for j, res in enumerate(r["results"], 1):
            mark = " *" if res.source in exp_sources else "  "
            print(f"  {j}.{mark}dist={res.distance:.3f}  {res.source}")
        print()


if __name__ == "__main__":
    main()

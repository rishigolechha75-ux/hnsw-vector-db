"""Score the filtered-search planner against fixed thresholds using measured QPS.

For every (dataset size, selectivity) row in reports/*/filtered.json we know the QPS
of both plans, so each policy's choice can be graded as chosen_qps / best_qps.
Note: the cost model's constants were fitted on these same rows (in-sample).

    python benchmarks/eval_planner.py
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vecdb.collection import choose_filtered_plan  # noqa: E402

POLICIES = {
    "cost model (default)": lambda m, n, ef: choose_filtered_plan(m, n, ef),
    "fixed count <= 10,000": lambda m, n, ef: choose_filtered_plan(m, n, ef, 10_000),
    "fixed selectivity < 0.1": lambda m, n, ef: "exact_filtered" if m / n < 0.1 else "hnsw_filtered",
    "always HNSW": lambda m, n, ef: "hnsw_filtered",
    "always exact": lambda m, n, ef: "exact_filtered",
}


def main():
    cases = []
    for path in sorted(glob.glob("reports/glove25_n*/filtered.json")):
        f = json.load(open(path))
        for r in f["rows"]:
            cases.append((f["n_base"], f["ef"], r))
    out = {"cases": len(cases), "note": "cost-model constants fitted on these rows", "policies": {}}
    for name, pol in POLICIES.items():
        ratios, optimal = [], 0
        for n, ef, r in cases:
            plan = pol(r["matching"], n, ef)
            got = r["exact_qps"] if plan == "exact_filtered" else r["hnsw_qps"]
            best = max(r["exact_qps"], r["hnsw_qps"])
            ratios.append(got / best)
            optimal += got == best
        out["policies"][name] = {"optimal_choices": optimal,
                                 "worst_ratio_to_best": min(ratios),
                                 "mean_ratio_to_best": sum(ratios) / len(ratios)}
    with open("reports/planner.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

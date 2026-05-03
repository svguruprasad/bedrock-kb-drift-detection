"""Validation experiment for the backward compatibility framework.

Generates synthetic response pairs with controlled drift levels
and validates that the framework correctly classifies them.

Run: python experiments/validate_framework.py
"""

import json
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from similarity import cosine_similarity, rouge_l, composite_score
from stats import run_drift_tests, classify_tier


def generate_vector(dim=1024, seed=None):
    """Generate a random unit vector."""
    if seed is not None:
        random.seed(seed)
    v = [random.gauss(0, 1) for _ in range(dim)]
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def perturb_vector(v, noise_level):
    """Add controlled noise to a vector. Higher noise = more drift."""
    noisy = [x + random.gauss(0, noise_level) for x in v]
    mag = sum(x * x for x in noisy) ** 0.5
    return [x / mag for x in noisy]


def perturb_text(text, swap_rate):
    """Swap words in text at a controlled rate. Higher rate = more structural drift."""
    words = text.split()
    synonyms = {
        "coverage": "protection", "includes": "covers", "policy": "plan",
        "claim": "request", "deductible": "excess", "damage": "loss",
        "standard": "typical", "rental": "temporary", "accident": "incident",
        "filing": "submitting", "comprehensive": "full", "exclusions": "exceptions",
    }
    result = []
    for w in words:
        lower = w.lower().strip(".,?!")
        if lower in synonyms and random.random() < swap_rate:
            replacement = synonyms[lower]
            if w[0].isupper():
                replacement = replacement.capitalize()
            result.append(replacement)
        else:
            result.append(w)
    return " ".join(result)


# Synthetic baseline responses (insurance domain)
BASELINE_RESPONSES = {
    "What does comprehensive coverage include?":
        "Comprehensive coverage includes protection against damage from events other than collision. This covers theft, vandalism, natural disasters, falling objects, and animal strikes. It does not cover mechanical breakdowns or normal wear and tear. The deductible applies to each claim filed under comprehensive coverage.",
    "How do I file a claim after an accident?":
        "To file a claim after an accident, contact your insurance provider within 24 hours. Provide the policy number, date and location of the accident, and a description of the damage. Take photographs of the damage and obtain a police report if applicable. An adjuster will be assigned to evaluate the claim within 3 to 5 business days.",
    "What is the standard deductible for collision coverage?":
        "The standard deductible for collision coverage ranges from $250 to $1,000. Most policyholders choose a $500 deductible, which balances monthly premium cost against out-of-pocket expense at claim time. Higher deductibles result in lower premiums. The deductible amount is subtracted from the claim payout.",
    "Are rental cars covered under my policy?":
        "Rental car coverage is an optional add-on to your policy. If you have rental reimbursement coverage, your policy pays for a rental car while your vehicle is being repaired after a covered claim. The daily limit and maximum duration vary by policy. Without this coverage, rental car expenses are your responsibility.",
    "What exclusions apply to flood damage?":
        "Standard auto insurance policies exclude flood damage from collision coverage. Flood damage is covered under comprehensive coverage if you have it. Homeowner policies typically exclude flood damage entirely. Separate flood insurance through the National Flood Insurance Program or a private insurer is required for property flood coverage.",
}


def run_experiment():
    """Run validation experiments at three drift levels."""
    experiments = [
        {"name": "No drift (same model, different invocations)", "noise": 0.005, "swap_rate": 0.0},
        {"name": "Minor drift (minor version update)", "noise": 0.005, "swap_rate": 0.10},
        {"name": "Moderate drift (major version update)", "noise": 0.02, "swap_rate": 0.25},
        {"name": "Major drift (model family change)", "noise": 0.08, "swap_rate": 0.45},
    ]

    weights = {"semantic": 0.40, "structural": 0.10, "factual": 0.50}
    threshold = 0.85
    alpha = 0.05
    k = 10  # repetitions

    all_results = {}

    for exp in experiments:
        print(f"\n{'=' * 70}")
        print(f"Experiment: {exp['name']}")
        print(f"  Embedding noise: {exp['noise']}")
        print(f"  Text swap rate: {exp['swap_rate']}")
        print(f"{'=' * 70}")

        query_scores = {}

        for query, baseline_text in BASELINE_RESPONSES.items():
            scores = []
            baseline_embedding = generate_vector(seed=hash(query) % 2**31)

            for rep in range(k):
                # Simulate candidate response
                candidate_text = perturb_text(baseline_text, exp["swap_rate"])
                candidate_embedding = perturb_vector(baseline_embedding, exp["noise"])

                sem = cosine_similarity(baseline_embedding, candidate_embedding)
                struct = rouge_l(baseline_text, candidate_text)
                fact = sem  # proxy
                score = composite_score(
                    sem, struct, fact,
                    weights["semantic"], weights["structural"], weights["factual"],
                )
                scores.append(score)

            query_scores[query] = scores
            mean = sum(scores) / len(scores)
            std = (sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)) ** 0.5
            print(f"  {mean:.4f} (std={std:.4f}) - {query[:55]}")

        results = run_drift_tests(query_scores, threshold, alpha)
        tier = classify_tier(results)

        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        print(f"\n  Tier: {tier}")
        print(f"  Passed: {passed}/{len(results)}, Failed: {failed}/{len(results)}")

        for r in results:
            status = "PASS" if r.passed else "FAIL"
            print(f"    [{status}] mean={r.mean_score:.4f} t={r.t_statistic:.2f} p={r.p_value:.4f}")

        all_results[exp["name"]] = {
            "tier": tier,
            "passed": passed,
            "failed": failed,
            "total": len(results),
            "details": [
                {
                    "query": r.query,
                    "mean": r.mean_score,
                    "std": r.std_dev,
                    "t_stat": r.t_statistic,
                    "p_value": r.p_value,
                    "passed": r.passed,
                }
                for r in results
            ],
        }

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    for name, res in all_results.items():
        print(f"  {name}: {res['tier']} ({res['passed']}/{res['total']} passed)")

    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_experiment()

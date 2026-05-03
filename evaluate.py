"""Model backward compatibility evaluation for Bedrock Knowledge Bases.

Usage:
    python evaluate.py --config config.yaml [--baseline-only] [--publish]
"""

import argparse
import json
import sys

import yaml

from bedrock_client import BedrockClient
from similarity import cosine_similarity, rouge_l, composite_score
from stats import run_drift_tests, classify_tier
from monitor import DriftMonitor


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_responses(
    client: BedrockClient,
    queries: list[str],
    kb_id: str,
    model_arn: str,
    repetitions: int,
) -> dict[str, list[dict]]:
    """Generate responses and embeddings for all queries."""
    results = {}
    for query in queries:
        responses = []
        for _ in range(repetitions):
            text = client.get_kb_response(query, kb_id, model_arn)
            embedding = client.get_embedding(text)
            responses.append({"text": text, "embedding": embedding})
        results[query] = responses
    return results


def compute_drift_scores(
    baseline: dict[str, list[dict]],
    candidate: dict[str, list[dict]],
    weights: dict[str, float],
) -> dict[str, list[float]]:
    """Compute composite drift scores for all query pairs."""
    query_scores = {}

    w_sem = weights["semantic"]
    w_struct = weights["structural"]
    w_fact = weights["factual"]

    for query in baseline:
        scores = []
        base_responses = baseline[query]
        cand_responses = candidate[query]

        for base, cand in zip(base_responses, cand_responses):
            sem = cosine_similarity(base["embedding"], cand["embedding"])
            struct = rouge_l(base["text"], cand["text"])
            # Factual consistency placeholder: use semantic similarity as proxy
            # In production, replace with NLI model scoring
            fact = sem
            score = composite_score(sem, struct, fact, w_sem, w_struct, w_fact)
            scores.append(score)

        query_scores[query] = scores

    return query_scores


def save_baseline(data: dict, path: str):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Baseline saved to {path}")


def load_baseline(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Model backward compatibility evaluation")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--baseline-only", action="store_true", help="Generate baseline only")
    parser.add_argument("--baseline-path", default="baseline.json", help="Baseline file path")
    parser.add_argument("--publish", action="store_true", help="Publish results to CloudWatch")
    args = parser.parse_args()

    config = load_config(args.config)
    client = BedrockClient(region=config["region"])

    queries = config["test_queries"]
    kb_id = config["knowledge_base_id"]
    reps = config["repetitions"]

    if args.baseline_only:
        print(f"Generating baseline with {reps} repetitions per query...")
        baseline = generate_responses(
            client, queries, kb_id, config["baseline_model_arn"], reps
        )
        save_baseline(baseline, args.baseline_path)
        return

    # Load baseline
    print(f"Loading baseline from {args.baseline_path}...")
    baseline = load_baseline(args.baseline_path)

    # Generate candidate responses
    print(f"Generating candidate responses with {reps} repetitions per query...")
    candidate = generate_responses(
        client, queries, kb_id, config["candidate_model_arn"], reps
    )

    # Compute drift scores
    print("Computing drift scores...")
    query_scores = compute_drift_scores(baseline, candidate, config["weights"])

    # Run statistical tests
    threshold = config["drift_threshold"]
    alpha = config["significance_level"]
    results = run_drift_tests(query_scores, threshold, alpha)
    tier = classify_tier(results)

    # Report
    print(f"\n{'=' * 70}")
    print(f"Model Backward Compatibility Report")
    print(f"{'=' * 70}")
    print(f"Baseline model: {config['baseline_model_arn'].split('/')[-1]}")
    print(f"Candidate model: {config['candidate_model_arn'].split('/')[-1]}")
    print(f"Threshold: {threshold}")
    print(f"Significance level: {alpha}")
    print(f"Repetitions per query: {reps}")
    print(f"Overall tier: {tier}")
    print(f"{'=' * 70}")

    passed = sum(1 for r in results if r.passed)
    failed = len(results) - passed
    print(f"Passed: {passed}/{len(results)}  Failed: {failed}/{len(results)}")
    print()

    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.mean_score:.4f} (std={r.std_dev:.4f}, t={r.t_statistic:.2f}, p={r.p_value:.4f})")
        print(f"         {r.query[:70]}")

    # Publish to CloudWatch
    if args.publish:
        print(f"\nPublishing to CloudWatch namespace: {config['cloudwatch_namespace']}")
        monitor = DriftMonitor(config["cloudwatch_namespace"], config["region"])
        monitor.publish_results(kb_id, results, tier)
        print("Published.")

    # Exit code based on tier
    if tier == "Red":
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

# Model Backward Compatibility for Amazon Bedrock Knowledge Bases

A quantitative framework for measuring whether a model migration preserves response quality in retrieval-augmented generation (RAG) systems.

## The problem

When you swap the foundation model in a RAG system, responses may change. Some changes improve quality. Others degrade it. This framework measures the difference and tells you whether the migration is safe.

## How it works

The framework computes a composite drift score from three metrics:

- **Semantic similarity** (cosine similarity of Titan Embeddings V2 vectors): Do the responses mean the same thing?
- **Structural similarity** (ROUGE-L F1): Do the responses use the same words and structure?
- **Factual consistency** (entailment scoring): Does the new response stay grounded in the source documents?

Statistical hypothesis testing (one-sample t-test with Bonferroni correction) separates meaningful drift from probabilistic noise. Results map to risk tiers:

| Tier | Condition | Action |
|------|-----------|--------|
| Green | All queries pass | Safe to migrate |
| Yellow | 1-5% of queries fail | Investigate failing queries |
| Red | More than 5% fail | Do not migrate |

## Quick start

```bash
pip install -r requirements.txt

# Generate baseline with current model
python evaluate.py --config config.yaml --baseline-only

# Evaluate candidate model against baseline
python evaluate.py --config config.yaml

# Evaluate and publish to CloudWatch
python evaluate.py --config config.yaml --publish
```

## Configuration

Edit `config.yaml`:

```yaml
knowledge_base_id: "your-kb-id"
baseline_model_arn: "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-20250514"
candidate_model_arn: "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-sonnet-4-20250514"

weights:
  semantic: 0.40
  structural: 0.10
  factual: 0.50

drift_threshold: 0.85
significance_level: 0.05
repetitions: 10
```

## Running tests

```bash
pytest tests/ -v
```

## Project structure

```
bedrock-kb-drift-detection/
├── config.yaml          # Configuration
├── evaluate.py          # Main evaluation script
├── similarity.py        # Cosine similarity, ROUGE-L, composite score
├── stats.py             # T-test, Bonferroni correction, tier classification
├── bedrock_client.py    # Amazon Bedrock API client
├── monitor.py           # CloudWatch metric publishing
├── requirements.txt     # Dependencies
├── tests/
│   └── test_all.py      # Unit and integration tests
└── README.md
```

## White paper

This implementation accompanies the white paper: "Measuring model backward compatibility in retrieval-augmented generation systems: A quantitative framework for safe model migration."

## License

MIT

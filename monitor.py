"""CloudWatch integration for drift monitoring."""

import boto3
from stats import TestResult


class DriftMonitor:
    def __init__(self, namespace: str, region: str = "us-east-1"):
        self.cloudwatch = boto3.client("cloudwatch", region_name=region)
        self.namespace = namespace

    def publish_results(self, kb_id: str, results: list[TestResult], tier: str):
        """Publish drift test results as CloudWatch custom metrics."""
        metric_data = []

        for result in results:
            metric_data.append(
                {
                    "MetricName": "DriftScore",
                    "Dimensions": [
                        {"Name": "KnowledgeBaseId", "Value": kb_id},
                    ],
                    "Value": result.mean_score,
                    "Unit": "None",
                }
            )

        # Publish tier as a numeric metric (Green=0, Yellow=1, Red=2)
        tier_value = {"Green": 0, "Yellow": 1, "Red": 2}.get(tier, 2)
        metric_data.append(
            {
                "MetricName": "MigrationRiskTier",
                "Dimensions": [
                    {"Name": "KnowledgeBaseId", "Value": kb_id},
                ],
                "Value": tier_value,
                "Unit": "None",
            }
        )

        # CloudWatch accepts max 1000 metrics per call
        for i in range(0, len(metric_data), 1000):
            batch = metric_data[i : i + 1000]
            self.cloudwatch.put_metric_data(
                Namespace=self.namespace,
                MetricData=batch,
            )

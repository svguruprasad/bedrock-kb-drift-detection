"""Amazon Bedrock client for RAG responses and embeddings."""

import json
import boto3


class BedrockClient:
    def __init__(self, region: str = "us-east-1"):
        self.agent_runtime = boto3.client("bedrock-agent-runtime", region_name=region)
        self.bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)

    def get_kb_response(
        self, question: str, kb_id: str, model_arn: str
    ) -> str:
        """Get a response from a Bedrock Knowledge Base."""
        response = self.agent_runtime.retrieve_and_generate(
            input={"text": question},
            retrieveAndGenerateConfiguration={
                "knowledgeBaseConfiguration": {
                    "knowledgeBaseId": kb_id,
                    "modelArn": model_arn,
                },
                "type": "KNOWLEDGE_BASE",
            },
        )
        return response["output"]["text"]

    def get_embedding(
        self, text: str, model_id: str = "amazon.titan-embed-text-v2:0"
    ) -> list[float]:
        """Get embedding vector from Amazon Titan Text Embeddings V2."""
        response = self.bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({"inputText": text}),
        )
        result = json.loads(response["body"].read())
        return result["embedding"]

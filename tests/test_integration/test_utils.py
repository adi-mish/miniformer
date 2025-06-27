import os
import json
import torch
import tempfile
from types import SimpleNamespace
from typing import List, Dict, Union, Optional, Any


class DummyTokenizer:
    """A simple tokenizer for testing purposes."""
    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size
    
    def encode(self, text, add_special_tokens=True):
        # Simple character-level encoding for testing
        return [ord(c) % self.vocab_size for c in text]


def create_jsonl_dataset(
    tmpdir: str,
    data: List[Dict[str, Any]],
    filename: str = "data.jsonl"
) -> str:
    """Create a temporary JSONL file with the given data."""
    filepath = os.path.join(tmpdir, filename)
    
    with open(filepath, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    return filepath


def create_test_datasets(tmpdir: str, task: str):
    """Create test datasets for a specific task."""
    if task == "language_modeling":
        train_data = [{"text": f"This is sample text {i}."} for i in range(20)]
        val_data = [{"text": f"This is validation text {i}."} for i in range(10)]
    elif task == "classification":
        train_data = [{"input": f"sample {i}", "label": i % 3} for i in range(20)]
        val_data = [{"input": f"val {i}", "label": i % 3} for i in range(10)]
    elif task == "regression":
        train_data = [{"input": f"feature {i}", "labels": i * 0.5} for i in range(20)]
        val_data = [{"input": f"val {i}", "labels": i * 0.5} for i in range(10)]
    else:
        raise ValueError(f"Unsupported task: {task}")
        
    train_path = create_jsonl_dataset(tmpdir, train_data, "train.jsonl")
    val_path = create_jsonl_dataset(tmpdir, val_data, "val.jsonl")
    
    return train_path, val_path


def verify_model_equivalence(
    model1: torch.nn.Module,
    model2: torch.nn.Module,
    input_tensor: torch.Tensor,
    tolerance: float = 1e-5
) -> bool:
    """
    Verify that two models produce the same output given the same input.
    
    Args:
        model1: First model
        model2: Second model
        input_tensor: Input tensor to test with
        tolerance: Tolerance for comparing outputs
        
    Returns:
        bool: True if the models produce equivalent outputs
    """
    model1.eval()
    model2.eval()
    
    with torch.no_grad():
        output1 = model1(input_tensor)
        output2 = model2(input_tensor)
        
    if isinstance(output1, tuple):
        output1 = output1[0]
    if isinstance(output2, tuple):
        output2 = output2[0]
        
    return torch.allclose(output1, output2, atol=tolerance)


def extract_metrics(trainer_logs: List[Dict]) -> Dict[str, float]:
    """Extract the latest metrics from trainer logs."""
    metrics = {}
    for log in reversed(trainer_logs):
        for key, value in log.items():
            if key not in metrics and isinstance(value, (int, float)):
                metrics[key] = value
                
    return metrics

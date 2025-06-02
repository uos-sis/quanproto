import torch
import pytest
from quanproto.metrics.compactness import local_explanation_size


def test_local_explanation_size_basic():
    # Test with normal values
    similarity_scores = torch.tensor([[2, 0.2], [0.05, 0.04], [1.1, 10.0]])
    output = local_explanation_size(similarity_scores, threshold=0.1)
    expected_output = torch.tensor(
        [1.0, 2.0, 2.0]
    )  # Two activations for the first, one for the second, one for the third
    assert torch.allclose(output, expected_output)


def test_local_explanation_size_all_zero():
    # Test with all zero values
    similarity_scores = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    output = local_explanation_size(similarity_scores, threshold=0.1)
    expected_output = torch.tensor([0.0, 0.0])  # No activations
    assert torch.allclose(output, expected_output)


def test_local_explanation_size_edge_case():
    # Test with scores exactly at the threshold
    similarity_scores = torch.tensor([[0.02, 0.2], [0.11, 0.1], [0.01, 0.1]])
    output = local_explanation_size(similarity_scores, threshold=0.1)
    expected_output = torch.tensor(
        [1.0, 2.0, 1.0]
    )  # Two activations for first, one for second, two for third
    assert torch.allclose(output, expected_output)


def test_local_explanation_size_single_value():
    # Test with a single value
    similarity_scores = torch.tensor([[0.2]])
    output = local_explanation_size(similarity_scores, threshold=0.1)
    expected_output = torch.tensor([1.0])  # One activation
    assert torch.allclose(output, expected_output)


if __name__ == "__main__":
    pytest.main()

"""
The prototype_model_interfaces module contains the PrototypeModelInterface class, which is an
abstract class that defines the methods that should be implemented by the prototype model classes
to ensure compatibility with the evaluation scripts.

Author: Philipp Schlinge
"""

from abc import ABC, abstractmethod

import torch


class PrototypeModelInterface(ABC):
    """
    The PrototypeModelInterface class is an abstract class that defines the methods that should be
    implemented by the prototype model classes to ensure compatibility with the evaluation scripts.
    """

    @abstractmethod
    def compile(self, coefs: dict | None, class_weights: list | None) -> None:
        """
        The compile method is primarily used to setup the weight parameters of the loss function.
        """

    @abstractmethod
    def warmup(self) -> None:
        """
        The warmup method is used to freeze the layers that should not be trained during the warmup
        phase.
        """

    @abstractmethod
    def joint(self) -> None:
        """
        The joint method is used to unfreeze all layers that should be trained during the joint
        phase.
        """

    @abstractmethod
    def fine_tune(self) -> None:
        """
        The fine_tune method is used to freeze all layers that should not be trained during the
        fine-tuning phase.
        """

    @abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        The predict method is used to make predictions on the input data.

        Returns:
            logits: The output of the model
        """

    @abstractmethod
    def explain(self, x: torch.Tensor) -> tuple:
        """
        The explain method is used to generate the prototype saliency maps.

        Returns:
            logits: The output of the model
            similarity_maps: The similarity maps of the prototype
            saliency_maps: The saliency maps of the prototype
        """

    @abstractmethod
    def get_prototypes(self, *args, **kwargs) -> torch.Tensor:
        """
        The get_prototypes method is used to get the prototype vectors.
        If the model does not have prototype vectors, then this method should return None.

        Returns:
            prototypes: The prototype vectors
        """

    @abstractmethod
    def global_explanation_size(self, epsilon: float) -> int:
        """
        The global_explanation_size method is used to get the size of the global explanation.

        Returns:
            global_explanation_size: The size of the global explanation
        """

    @abstractmethod
    def classification_sparsity(self, epsilon):
        """
        The classification_sparsity method is used to get the classification sparsity.

        Returns:
            classification_sparsity: The classification sparsity
        """

    @abstractmethod
    def negative_positive_reasoning_ratio(self, epsilon):
        """
        The negative_positive_reasoning_ratio method is used to get the negative positive reasoning
        ratio.

        Returns:
            negative_positive_reasoning_ratio: The negative positive reasoning ratio
        """

"""
This file contains the ExplanationInterface class, which is an abstract class
that defines the interface for the explanation classes in the QuanProto library.
An explanation class is used as a wrapper for a prototype model. 

The idea is that it inherits from the Model class and extends the functionality
of the model by adding methods to compute saliency maps for the input data.

Author:
    Philipp Schlinge
"""

import torch
from abc import ABC, abstractmethod


class ExplanationInterface(ABC):
    @abstractmethod
    def canonize(self, *args, **kwargs) -> None:
        """
        This method is used to canonize the model so it can be used for the
        explanation method.

        Note:
            This method is typically used after the state dictionary of the
            model has been loaded. As it often changes the names of the layers.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        pass

    @abstractmethod
    def saliency_maps(self, *args, **kwargs) -> torch.Tensor:
        """
        This method is used to compute the saliency maps for the input data.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            torch.Tensor: The saliency maps for the input data.
        """
        pass

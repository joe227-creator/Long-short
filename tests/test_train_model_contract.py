"""Regression tests for model architecture compatibility details."""

import unittest

from train import LSTMModel


class LSTMModelContractTests(unittest.TestCase):
    def test_unused_input_layer_norm_is_checkpoint_compatibility_only(self):
        model = LSTMModel(input_dim=3)
        self.assertFalse(model.layer_norm.weight.requires_grad)
        self.assertFalse(model.layer_norm.bias.requires_grad)


if __name__ == "__main__":
    unittest.main()

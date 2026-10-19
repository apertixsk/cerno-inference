"""Loads the inference wrapper for a Cerno checkpoint (the decider-style causal LM readout; an encoder variant if the checkpoint declares one)."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "decider"))


def load_engine(path, temperature=None):
    if os.path.exists(os.path.join(path, "cerno_encoder.json")):
        from encoder.model import EncoderSystemOne
        return EncoderSystemOne(path, temperature=temperature)
    from decider.infer import Decider
    return Decider(path, temperature=temperature)

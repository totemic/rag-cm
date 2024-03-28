from typing import Any, Callable
import logging
logger = logging.getLogger(__name__)


# import colbert.modeling.checkpoint as CP
# from colbert.modeling.colbert import ColBERT

# original_class = CP.Checkpoint
# class WrappedCheckpoint(original_class):
#     def __init__(self, name, colbert_config=None, verbose:int = 3):


# from colbert.modeling.checkpoint import Checkpoint

# original_checkpoint_init = Checkpoint.__init__

# def patched_checkpoint_init(self, name, colbert_config=None, verbose:int = 3):
#     original_checkpoint_init(self, name, colbert_config)

# Checkpoint.__init__ = patched_checkpoint_init

# from colbert.modeling.hf_colbert import class_factory

from colbert.data import Collection
import colbert.modeling.hf_colbert as hf_colbert

import sys

hf_colbert_class_factory = hf_colbert.class_factory
model_cache: dict[str, Any] = {}
tokenizer_cache: dict[str, Any] = {}

def new_class_factory(name_or_path: str):
    factory = hf_colbert_class_factory(name_or_path)
    factory.from_pretrained_original = factory.from_pretrained
    factory.raw_tokenizer_from_pretrained_original = factory.raw_tokenizer_from_pretrained

    def from_pretrained_patched(name_or_path: str, colbert_config: Any) -> Any:
        if name_or_path in model_cache:
            return model_cache[name_or_path]
        model: Any = factory.from_pretrained_original(name_or_path, colbert_config)
        if model is not None:
            model_cache[name_or_path] = model
        return model

    def raw_tokenizer_from_pretrained(name_or_path:str) -> Any:
        if name_or_path in tokenizer_cache:
            return tokenizer_cache[name_or_path]
        tokenizer: Any = factory.raw_tokenizer_from_pretrained_original(name_or_path)
        if tokenizer is not None:
            tokenizer_cache[name_or_path] = tokenizer
        return tokenizer

    factory.from_pretrained = from_pretrained_patched
    factory.raw_tokenizer_from_pretrained = raw_tokenizer_from_pretrained
    return factory

# https://medium.com/@chipiga86/python-monkey-patching-like-a-boss-87d7ddb8098e
def uncache(exclude: list[str]):
    """Remove package modules from cache except excluded ones.
    On next import they will be reloaded.
    
    Args:
        exclude (iter<str>): Sequence of module paths.
    """
    pkgs: list[str] = []
    for mod in exclude:
        pkg: str = mod.split('.', 1)[0]
        pkgs.append(pkg)

    to_uncache: list[str] = []
    for mod in sys.modules:
        if mod in exclude:
            continue

        if mod in pkgs:
            to_uncache.append(mod)
            continue

        for pkg in pkgs:
            if mod.startswith(pkg + '.'):
                to_uncache.append(mod)
                break

    for mod in to_uncache:
        del sys.modules[mod]

# def patch_colbert

def patch_colbert_model_loader_be_carefull():
    hf_colbert.class_factory = new_class_factory
    uncache(['colbert.modeling.hf_colbert'])
    logger.warn('Patching colbert.modeling.hf_colbert')

def patch_colbert_collection_cast_be_carefull():
    # Monkey patch the cast function
    # TODO: remove once this is merged https://github.com/stanford-futuredata/ColBERT/pull/330
    # we can't execute the code here, this function is just collecting all the patches in one file for reference
    logger.warn('Patching colbert.data.Collection')
"""Corpus labels with a separately identified native source-readback contract."""
import hashlib
from pathlib import Path

from obruxo_data.render.base import FULL_CORPUS_RENDERER_ID
from obruxo_data.vital.full_contract import FullPresetContract

from .contracts import TrainingContract
from .corpus_inputs import CorpusInputCodec


class CorpusNativeCodec:
    def __init__(self, asset_bank, lfo_bank):
        self.parent = CorpusInputCodec(asset_bank, lfo_bank)
        self.schema = FullPresetContract(renderer_id=FULL_CORPUS_RENDERER_ID)
        self.catalog = self.parent.catalog
        self.contract = TrainingContract.create(self.schema, self.catalog, tail_seconds=4.)

    def __getattr__(self, name):
        return getattr(self.parent, name)

    @property
    def codec_id(self):
        from obruxo_data.hashing import canonical_sha256
        return canonical_sha256(self.to_dict())

    def to_dict(self):
        spec = self.parent.to_dict()
        spec.update(codec="canonical_corpus_native_v1", renderer_id=self.contract.renderer_id, contract_id=self.contract.contract_id,
                    corpus_native_policy=self.schema.spec["corpus_native_policy"],
                    corpus_codec_source_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                    scope=spec["scope"] + "_with_original_source_native_roundtrip_readback")
        return spec

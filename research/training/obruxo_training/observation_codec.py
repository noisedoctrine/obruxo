"""A separately identified longer-observation contract over the unchanged native codec."""
from dataclasses import replace
import hashlib
from pathlib import Path

from .contracts import TrainingContract
from .corpus_codec import CorpusNativeCodec
from .observation_bank import PROFILE, observation_bank_document


class ObservationCodec(CorpusNativeCodec):
    def __init__(self, asset_bank, lfo_bank):
        super().__init__(asset_bank, lfo_bank)
        self.contract = TrainingContract.create(self.schema, self.catalog, tail_seconds=32., performance_profile=PROFILE)

    def to_dict(self):
        spec = super().to_dict()
        spec.update(codec="canonical_corpus_observations_v1", performance_bank=observation_bank_document(),
                    observation_codec_source_sha256_lf=hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest())
        return spec


def extend_observations(pool):
    from .corpus_pool import CorpusPool
    if not isinstance(pool, CorpusPool) or type(pool.codec) is not CorpusNativeCodec:
        raise ValueError("observation extension requires an original native corpus source pool")
    codec = ObservationCodec(pool.codec.base.bank, pool.codec.lfo_bank.to_dict())
    return replace(pool, codec=codec)

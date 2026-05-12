from dataclasses import dataclass
import sentencepiece as spm

@dataclass
class Tokenizer:
    model_path: str

    def __post_init__(self):
        self.sp = spm.SentencePieceProcessor()
        ok = self.sp.Load(self.model_path)
        if not ok:
            raise FileNotFoundError(f"Could not load tokenizer model at {self.model_path}")

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        return list(self.sp.EncodeAsIds(text))

    def decode(self, ids: list[int]) -> str:
        return self.sp.DecodeIds([int(x) for x in ids])

    @property
    def bos_id(self) -> int:
        return int(self.sp.bos_id())

    @property
    def eos_id(self) -> int:
        return int(self.sp.eos_id())

    @property
    def pad_id(self) -> int:
        return int(self.sp.pad_id())

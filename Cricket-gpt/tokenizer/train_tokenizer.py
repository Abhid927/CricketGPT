import os
import sentencepiece as spm

def main():
    vocab_size = int(os.environ.get("VOCAB_SIZE", "8000"))
    model_prefix = os.environ.get("MODEL_PREFIX", "tokenizer/cricket_bpe")
    input_file = os.environ.get("INPUT_FILE", "data/processed/train.txt")

    # SentencePiece trains on a text file directly.
    # BPE is a good default for small LMs.
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        bos_id=1,  # keep 0 as <unk>
        eos_id=2,
        pad_id=3,
        unk_id=0,
        user_defined_symbols=["<|user|>", "<|assistant|>", "<|match|>"]
    )

    print(f"Saved: {model_prefix}.model and {model_prefix}.vocab")

if __name__ == "__main__":
    main()

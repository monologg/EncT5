from enc_t5 import EncT5Tokenizer


def test_enc_t5_tokenizer():
    tokenizer = EncT5Tokenizer.from_pretrained("patrickvonplaten/t5-tiny-random")
    assert tokenizer.bos_token == "<s>"
    assert tokenizer.bos_token_id == 32100

    assert tokenizer.vocab_size == 32100
    assert len(tokenizer.get_vocab()) == 32101

    tokens = tokenizer.tokenize("<s> This is a test code </s> ")
    assert tokens == ["<s>", "▁This", "▁is", "▁", "a", "▁test", "▁code", "</s>"]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    assert ids == [32100, 100, 19, 3, 9, 794, 1081, 1]

    back_tokens = tokenizer.convert_ids_to_tokens(ids)
    assert back_tokens == ["<s>", "▁This", "▁is", "▁", "a", "▁test", "▁code", "</s>"]

    assert tokenizer("This is a test code")["input_ids"] == [32100, 100, 19, 3, 9, 794, 1081, 1]
    assert tokenizer("This is a test code", "That is a test code")["input_ids"] == [
        32100,
        100,
        19,
        3,
        9,
        794,
        1081,
        1,
        466,
        19,
        3,
        9,
        794,
        1081,
        1,
    ]

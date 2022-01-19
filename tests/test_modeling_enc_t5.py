from enc_t5 import EncT5ForSequenceClassification


def test_load_model_enc_t5():
    EncT5ForSequenceClassification.from_pretrained("patrickvonplaten/t5-tiny-random")

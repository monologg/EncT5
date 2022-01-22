# EncT5

(Unofficial) Pytorch Implementation of `EncT5`: [Fine-tuning T5 Encoder for Non-autoregressive Tasks](https://arxiv.org/abs/2110.08426)

## About

<div align="center">
    <img src="https://user-images.githubusercontent.com/28896432/150393562-a0b3af15-a1f5-43a4-af6f-0954fafb2458.png" width="600px">
</div>

- Finetune T5 model for classification & regression by **only using the encoder layers**.
- Implemented of [`Tokenizer`](./enc_t5/tokenization_enc_t5.py) and [`Model`](./enc_t5/modeling_enc_t5.py) for EncT5.
- Add **BOS Token (`<s>`)** for tokenizer, and use this token for classification & regression.
  - Need to resize embedding as vocab size is changed. (`model.resize_token_embeddings()`)

## Requirements

> Highly recommend to use the same version of `transformers`.

```
transformers==4.15.0
torch==1.8.1
sentencepiece==0.1.96
datasets==1.17.0
scikit-learn==0.24.2
```

## Finetune on GLUE

- Use `Huggingface Transformers Trainer` for finetuning GLUE Task.
- See more details in [Trainer Guideline](https://github.com/huggingface/transformers/blob/e03544a13804a32ff12afff98c8e60faa0fdc282/examples/pytorch/README.md) and [Trainer Documentation](https://huggingface.co/docs/transformers/v4.15.0/en/main_classes/trainer).

### Setup

- Use [`T5 1.1 base`](https://huggingface.co/google/t5-v1_1-base) for finetuning.
- Evaluate on TPU. See [`run_glue_tpu.sh`](./scripts/run_glue_tpu.sh) for more details.
- Use **`AdamW`** optimizer instead of `Adafactor`.
- Check best checkpoint on every epoch by using `EarlyStoppingCallback`.

### Results

| Task  | Metric      | Result (Paper) | Result (Implementation) |
| :---- | ----------- | :------------: | :---------------------: |
| CoLA  | Matthew     |      53.1      |                         |
| SST-2 | Acc         |      94.0      |                         |
| MRPC  | F1/Acc      |   91.5/88.3    |                         |
| STS-B | PCC/SCC     |   80.5/79.3    |                         |
| QQP   | F1/Acc      |   72.9/89.8    |                         |
| MNLI  | Mis/Matched |   88.0/86.7    |                         |
| QNLI  | Acc         |      93.3      |                         |
| RTE   | Acc         |      67.8      |                         |

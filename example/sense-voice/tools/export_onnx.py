#!/usr/bin/env python3

"""
Modified from https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/sense-voice/export-onnx.py
Thanks k2-fsa.
"""

import os
from typing import Any, Dict, Tuple
import onnx, onnxsim
import torch
from model import SenseVoiceSmall

INPUT_LEN = 124

def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)

def encoder_forward(
    self,
    xs_pad: torch.Tensor,
    ilens: torch.Tensor,
):
    """Embed positions in tensor."""
    masks = sequence_mask(ilens, maxlen=INPUT_LEN+4, device=ilens.device)[:, None, :]

    xs_pad *= self.output_size() ** 0.5

    xs_pad = self.embed(xs_pad)

    # forward encoder1
    for layer_idx, encoder_layer in enumerate(self.encoders0):
        encoder_outs = encoder_layer(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

    for layer_idx, encoder_layer in enumerate(self.encoders):
        encoder_outs = encoder_layer(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

    xs_pad = self.after_norm(xs_pad)

    # forward encoder2
    olens = masks.squeeze(1).sum(1).int()

    for layer_idx, encoder_layer in enumerate(self.tp_encoders):
        encoder_outs = encoder_layer(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

    xs_pad = self.tp_norm(xs_pad)
    return xs_pad, olens

def modified_forward(
    self,
    x: torch.Tensor,
    x_length: torch.Tensor,
    language: torch.Tensor,
    text_norm: torch.Tensor,
):
    """
    Args:
      x:
        A 3-D tensor of shape (N, T, C) with dtype torch.float32
      x_length:
        A 1-D tensor of shape (N,) with dtype torch.int32
      language:
        A 1-D tensor of shape (N,) with dtype torch.int32
        See also https://github.com/FunAudioLLM/SenseVoice/blob/a80e676461b24419cf1130a33d4dd2f04053e5cc/model.py#L640
      text_norm:
        A 1-D tensor of shape (N,) with dtype torch.int32
        See also https://github.com/FunAudioLLM/SenseVoice/blob/a80e676461b24419cf1130a33d4dd2f04053e5cc/model.py#L642
    """
    language_query = self.embed(language).unsqueeze(1)
    text_norm_query = self.embed(text_norm).unsqueeze(1)

    event_emo_query = self.embed(torch.LongTensor([[1, 2]])).repeat(x.size(0), 1, 1)

    x = torch.cat((language_query, event_emo_query, text_norm_query, x), dim=1)
    x_length += 4

    encoder_out, encoder_out_lens = self.encoder(x, x_length)
    if isinstance(encoder_out, tuple):
        encoder_out = encoder_out[0]

    ctc_logits = self.ctc.ctc_lo(encoder_out)

    return ctc_logits

def load_cmvn(filename) -> Tuple[str, str]:
    neg_mean = None
    inv_stddev = None

    with open(filename) as f:
        for line in f:
            if not line.startswith("<LearnRateCoef>"):
                continue
            t = line.split()[3:-1]

            if neg_mean is None:
                neg_mean = ",".join(t)
            else:
                inv_stddev = ",".join(t)

    return neg_mean, inv_stddev


def generate_tokens(params):
    sp = params["tokenizer"].sp
    with open("tokens.txt", "w", encoding="utf-8") as f:
        for i in range(sp.vocab_size()):
            f.write(f"{sp.id_to_piece(i)} {i}\n")

    os.system("head tokens.txt; tail -n200 tokens.txt")


def display_params(params):
    print("----------params----------")
    print(params)

    print("----------frontend_conf----------")
    print(params["frontend_conf"])

    os.system(f"cat {params['frontend_conf']['cmvn_file']}")

    print("----------config----------")
    print(params["config"])

    os.system(f"cat {params['config']}")


def main():
    # model, params = SenseVoiceSmall.from_pretrained(model="iic/SenseVoiceSmall", device="cpu")
    model, params = SenseVoiceSmall.from_pretrained(model="../models/SenseVoiceSmall", device="cpu")

    generate_tokens(params)

    model.__class__.forward = modified_forward
    model.encoder.__class__.forward = encoder_forward
    
    x = torch.randn(1, INPUT_LEN, 560, dtype=torch.float32)
    x_length = torch.tensor([100], dtype=torch.int32)
    language = torch.tensor([3], dtype=torch.int32)
    text_norm = torch.tensor([15], dtype=torch.int32)

    opset_version = 13
    filename = "model_encoder_test.onnx"
    torch.onnx.export(
        model,
        (x, x_length, language, text_norm),
        filename,
        opset_version=opset_version,
        input_names=["x", "x_length", "language", "text_norm"],
        output_names=["logits"],
    )
    # sim_model,_ = onnxsim.simplify(filename)
    # onnx.save(sim_model, filename)


if __name__ == "__main__":
    torch.manual_seed(20250717)
    main()

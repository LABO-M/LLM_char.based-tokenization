# GPT-2の実装 code from https://github.com/karpathy/makemore/tree/master
# https://github.com/karpathy/makemore/blob/master/makemore.py
"""
MIT License

Copyright (c) 2022 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset, DataLoader
# -----------------------------------------------------------------------------

@dataclass
class ModelConfig: #hyperparameters for the model
    block_size: int = None # length of the input sequences of integers
    vocab_size: int = None # the input integers are in range [0 .. vocab_size -1]
    # parameters below control the sizes of each model slightly differently
    n_layer: int = 12
    n_embd: int = 768
    #n_embd2: int = 64
    n_head: int = 12
    dropout: float = 0.1

# -----------------------------------------------------------------------------
# Transformer Language Model (*exactly* as used in GPT-2)

class NewGELU(nn.Module): #GELU activation function(ReLUの改良版)
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module): #multi-head masked self-attention layer with a projection at the end
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding size must be divisible by number of heads"
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head  # ヘッドごとの次元数
        self.scale = math.sqrt(self.head_dim)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)  # クエリ・キー・バリュー
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)      # 出力プロジェクション

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, return_att=False):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).chunk(3, dim=2)

        # 各ヘッドに分割
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # スケール付きドットプロダクトアテンション
        att = (q @ k.transpose(-2, -1)) / self.scale
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v

        # ヘッド次元を結合して出力
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)

        if return_att:
            return y,att
        return y

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            NewGELU(),
            nn.Dropout(config.dropout),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))  # 残差接続
        x = x + self.mlp(self.ln_2(x))  # 残差接続
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size

        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "wpe": nn.Embedding(config.block_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 層数をconfigに依存
            "ln_f": nn.LayerNorm(config.n_embd),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"トランスフォーマーの総パラメータ数: {n_params/1e6:.2f}M")

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"シーケンス長 {t} はブロックサイズ {self.block_size} を超えています"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = tok_emb + pos_emb

        attentions = []
        for block in self.transformer.h:
            x, att = block.attn(block.ln_1(x), return_att=True)
            attentions.append(att)
            x = block.mlp(block.ln_2(x))

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, attentions


    def generate(self, token_indexes, max_new_tokens, tokenizer, temperature=1.0, top_k=None):
        """
        token_indexes:(batch_size, sequence_length)
        max_new_tokens:生成するトークンの最大数
        temperature:出力分布をスケールするためのパラメータ
        top_k:確率上位Kこのトークンからサンプリングする場合の値
        """
        for _ in range(max_new_tokens):
            #入力シーケンスがbrock_sizeを超えた場合はトランケート
            token_indexes = token_indexes[:, -self.block_size:]

            #モデルの予測
            logits, _ = self(token_indexes)

            #現在の時刻のロジット
            logits = logits[:, -1, :]

            #温度付きサンプリング
            if temperature != 1.0:
                logits = logits / temperature

            #top_kサンプリング
            if top_k is not None:
                values, indices = torch.topk(logits, top_k,)
                logits = torch.zeros_like(logits).scatter(1, indices, values)

            #確率分布を計算
            probs = F.softmax(logits, dim=-1)

            #次のトークンをサンプリング
            next_token = torch.multinomial(probs, num_samples=1)

            #生成されたトークンを入力シーケンスに追加
            token_indexes = torch.cat([token_indexes, next_token], dim=1)

            sentence = tokenizer.decode(token_indexes[0].tolist())

        return sentence

    def compute_loss_for_sentence(self, sentence_tokens, device):
        """
        Computes the loss for a given sentence.
        Args:
        sentence_tokens: List of tokenized integers for the sentence.
        tokenizer: Tokenizer object to convert tokens to text.

        Returns:
        loss: The average loss value for the given sentence.
        """
        idx = torch.tensor(sentence_tokens, dtype=torch.long).unsqueeze(0).to(device)
        targets = idx.clone()  # In language modeling, target is usually the same sequence shifted by 1.
        targets = targets.clone()

        # Shift targets for next-token prediction
        targets[:, :-1] = targets[:, 1:]
        targets[:, -1] = -1  # Ignore the last token for loss calculation

        with torch.no_grad():
            _, loss = self.forward(idx, targets)

        return loss.item()

    def is_grammatically_correct(self, sentence_tokens, device, threshold=5.0):
        """
        Determines if a sentence is grammatically correct based on model's loss.
            Args:
            sentence_tokens: List of tokenized integers for the sentence.
            tokenizer: Tokenizer object to convert tokens to text.
            threshold: Loss threshold below which the sentence is considered grammatically correct.

        Returns:
            bool: True if grammatically correct, False otherwise.
        """
        loss = self.compute_loss_for_sentence(sentence_tokens, device)
        return loss < threshold  # Threshold can be adjusted based on empirical results

    def correct_sentence(self, input_tokens, max_new_tokens=50, temperature=1.0):
        """
        文法的誤り訂正タスクに特化した生成メソッド。
        Args:
            input_tokens (torch.Tensor): 文法的誤りを含む入力文のトークンシーケンス。
            max_new_tokens (int): 訂正後に生成する最大トークン数。
            temperature (float): 温度スケーリングパラメータ。

        Returns:
            torch.Tensor: 新しく生成されたトークンのシーケンス。
        """
        self.eval()  # 推論モード
        device = input_tokens.device

        # 初期入力シーケンス
        generated = input_tokens  # 文法誤りのある文
        input_length = input_tokens.size(1)  # 入力文の長さを記録

        for _ in range(max_new_tokens):
            # 入力シーケンスからロジットを取得
            input_seq = generated[:, -self.block_size:]  # ブロックサイズでトランケート
            logits, _ = self(input_seq)

            # 最後のトークンのロジットを取得
            logits = logits[:, -1, :]

            # 温度スケーリング
            logits = logits / temperature

            # 確率分布を計算して次のトークンをサンプリング
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 終了トークンが生成された場合は終了
            if next_token.item() == self.transformer.wte.num_embeddings - 1:  # 終了トークンを定義
                break

            # 生成トークンをシーケンスに追加
            generated = torch.cat((generated, next_token), dim=1)

        # 新しく生成されたトークンのみを返す
        new_tokens = generated[:, input_length:]
        return new_tokens


class BenchmarkDataset(Dataset):
    def __init__(self, tokenizer, texts, labels):
        """
        ベンチマーク用データセット。
        Args:
            tokenizer (Tokenizer): トークナイザーオブジェクト
            texts (List[str]): 入力テキストのリスト
            labels (List[int]): ラベル（分類タスク用）
        """
        self.inputs = [tokenizer.encode(text, eot=True, return_tensors="pt").squeeze(0) for text in texts]
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

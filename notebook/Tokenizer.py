#Tokenizer
#１文字１トークン
import os
import sys
import numpy as np
import torch

class Tokenizer:
    def __init__(self, chars):
        self.str_to_idx = dict()
        self.str_to_idx['<|endoftext|>'] = 0
        # utf-8
        for i in range(256):
            if f'<utf8_{i}>' not in self.str_to_idx:
                self.str_to_idx[f'<utf8_{i}>'] = len(self.str_to_idx)
        for char in chars:
            self.str_to_idx[char] = len(self.str_to_idx)
        self.idx_to_str = dict()
        for key, value in self.str_to_idx.items():
            self.idx_to_str[value] = key

    def encode(self, text, eot=False, return_tensors=None):
        """
        Textをトークン化して数値リストまたはテンソルとして返す
        Args:
            text (str): トークン化するテキスト
            eot (bool): 最後にEnd-of-Textトークンを追加するか
            return_tensors (str): "pt"ならTensorを返す, Noneならリストを返す
        """
        result = []
        for char in text:
            if char not in self.str_to_idx:
                utf_8_num = list(char.encode("utf-8"))
                for num in utf_8_num:
                    result.append(self.str_to_idx[f'<utf8_{num}>'])
            else:
                result.append(self.str_to_idx[char])
        if eot:
            result.append(self.str_to_idx['<|endoftext|>'])

        # return_tensorsオプションに応じて形式を変える
        if return_tensors == "pt":
            return torch.tensor(result, dtype=torch.long)
        return result

    def decode(self, tokens, skip_special_tokens=False):
        """
        トークンIDのリストをデコードしてテキストに変換する関数。

        Args:
            tokens (List[int]): トークンIDのリスト。
            skip_special_tokens (bool): 特殊トークンをデコード結果から除外するか。

        Returns:
            str: デコードされたテキスト。
        """
        decoded_with_utf_token = [self.idx_to_str.get(token, f"<unk_{token}>") for token in tokens]
        decoded_postprocess_utf = []
        utf_tokens = []

        for token in decoded_with_utf_token:
            if skip_special_tokens and token.startswith("<|") and token.endswith("|>"):
                continue

            if token.startswith("<utf8_"):
                try:
                    utf_num = int(token.replace("<utf8_", "").replace(">", ""))
                    utf_tokens.append(utf_num)
                except ValueError:
                    decoded_postprocess_utf.append(f"<invalid_utf_token:{token}>")
            else:
                if utf_tokens:
                    try:
                        decoded_postprocess_utf.append(bytes(utf_tokens).decode("utf-8", errors="replace"))
                    except UnicodeDecodeError:
                        decoded_postprocess_utf.append(f"<decode_error:{utf_tokens}>")
                    utf_tokens = []
                decoded_postprocess_utf.append(token)

        # 残ったUTFトークンを処理
        if utf_tokens:
            try:
                decoded_postprocess_utf.append(bytes(utf_tokens).decode("utf-8", errors="replace"))
            except UnicodeDecodeError:
                decoded_postprocess_utf.append(f"<decode_error:{utf_tokens}>")
            utf_tokens = []

        return "".join(decoded_postprocess_utf)

    def decode_with_utf(self, tokens):
        return "".join([self.idx_to_str[token] for token in tokens])

    def __call__(self, text, eot=False):
        return self.encode(text, eot)

#tokenizer = Tokenizer(unique_chars_in_train_text) # Tokenizerの初期化、一般的にはByte Pair EncodingやUnigram Language Modelなどを活用してTokenizerを実装する
#text = 'I am fine, thank you.'
#print(tokenizer.encode(text))

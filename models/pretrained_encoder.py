import torch
import torch.nn as nn
from typing import Union, List
from numpy import ndarray
from torch import Tensor, device
import numpy as np
from tqdm.autonotebook import trange

from transformers import AutoModel, AutoTokenizer, AutoConfig
from models.utils import batch_to_device


class PretrainedSentenceEncoder(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str,
        truncate: int = 128,
        pooling_method: str = "mean_pooling",
        normalize_embeddings: bool = True,
        num_layers = None
    ):
        super().__init__()
        
        if num_layers is None or num_layers < 0:        
            self.encoder = AutoModel.from_pretrained(pretrained_model_name)
        else:
            model_config = AutoConfig.from_pretrained(pretrained_model_name)
            model_config.update({"num_hidden_layers": num_layers})
            self.encoder = AutoModel.from_pretrained(pretrained_model_name, config=model_config)

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        self.truncate = truncate
        assert pooling_method in ["mean_pooling", "cls"]
        self.pooling_method = pooling_method
        self.normalize_embeddings = normalize_embeddings

    @property
    def hidden_size(self):
        return self.encoder.config.hidden_size

    def _get_pooled_embedding(self, encoded, attention_mask):

        if self.pooling_method == "cls":
            embeddings = encoded.last_hidden_state[:, 0, :]
        elif self.pooling_method == "mean_pooling":
            token_embeddings = encoded.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1)
                .expand(encoded.last_hidden_state.size())
                .float()
            )
            embeddings = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        if self.normalize_embeddings:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings

    def encode(self, sentences: List[str], device: str) -> List[Tensor]:

        tokenized = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.truncate,
        )

        return self.encode_tokenized_input(tokenized, device)

    def encode_tokenized_input(self, tokenized_sentences, device: str):
        tokenized_sentences = batch_to_device(tokenized_sentences, device)
        encoded = self.encoder(**tokenized_sentences)
        # use the cls or pooled representation
        pooled_embeddings = self._get_pooled_embedding(
            encoded, tokenized_sentences["attention_mask"]
        )
        return pooled_embeddings

    def batch_encode(
        self,
        sentences: Union[str, List[str]],
        device: str = "cpu",
        batch_size: int = 16,
        convert_to_numpy: bool = False,
    ) -> Union[List[Tensor], ndarray, Tensor]:

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches"):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            embeddings = self.encode(sentences=sentences_batch, device=device)
            all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_numpy:
            all_embeddings = np.asarray(
                [emb.detach().cpu().numpy() for emb in all_embeddings]
            )

        return all_embeddings


def _test_pretrained_encoder():
    pretrained_model_name = "bert-base-uncased"
    device = "cuda"
    model = PretrainedSentenceEncoder(pretrained_model_name=pretrained_model_name, num_layers=3)
    print(model)
    input = ["hello world", "great"]
    model.to(device)
    embeddings = model.encode(input, device=device)
    print(embeddings)


if __name__ == "__main__":
    _test_pretrained_encoder()

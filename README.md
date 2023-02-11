# PyTorch-SentenceEmbedding


https://arxiv.org/pdf/1908.10084.pdf


## stsb 

bert baseline no finetune
cls vector

Cosine-Similarity :     Pearson: 0.1718 Spearman: 0.2030
Manhattan-Distance:     Pearson: 0.2087 Spearman: 0.2078
Euclidean-Distance:     Pearson: 0.1989 Spearman: 0.2030
Dot-Product-Similarity: Pearson: 0.1718 Spearman: 0.2030

mean pooling
Cosine-Similarity :     Pearson: 0.4791 Spearman: 0.4729
Manhattan-Distance:     Pearson: 0.4809 Spearman: 0.4740
Euclidean-Distance:     Pearson: 0.4800 Spearman: 0.4729
Dot-Product-Similarity: Pearson: 0.4791 Spearman: 0.472


bert finetuned on nli


Cosine-Similarity :     Pearson: 0.2777 Spearman: 0.2989
Manhattan-Distance:     Pearson: 0.3113 Spearman: 0.3064
Euclidean-Distance:     Pearson: 0.3006 Spearman: 0.2989
Dot-Product-Similarity: Pearson: 0.2777 Spearman: 0.2989

mean pooling 
https://wandb.ai/yangyutu/sentence_bert_finetune/runs/yrd0t2sk?workspace=user-yangyutu
after 1 epoch
Cosine-Similarity :     Pearson: 0.5087 Spearman: 0.5054
Manhattan-Distance:     Pearson: 0.5143 Spearman: 0.5086
Euclidean-Distance:     Pearson: 0.5110 Spearman: 0.5054
Dot-Product-Similarity: Pearson: 0.5087 Spearman: 0.5054


after 2 epochs
Cosine-Similarity :     Pearson: 0.4630 Spearman: 0.4617
Manhattan-Distance:     Pearson: 0.4697 Spearman: 0.4625
Euclidean-Distance:     Pearson: 0.4681 Spearman: 0.4617
Dot-Product-Similarity: Pearson: 0.4630 Spearman: 0.4617

official baseline 
sentence-transformers/bert-base-nli-mean-tokens
Cosine-Similarity :     Pearson: 0.7415 Spearman: 0.7698
Manhattan-Distance:     Pearson: 0.7744 Spearman: 0.7702
Euclidean-Distance:     Pearson: 0.7734 Spearman: 0.7698
Dot-Product-Similarity: Pearson: 0.7415 Spearman: 0.7698


contrastive learning

mean pooling without length normalization

Cosine-Similarity :█████Pearson: 0.8412█Spearman: 0.8489███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊               | 44/47 [00:00<00:00, 88.33it/s]
Manhattan-Distance:█████Pearson: 0.8328█Spearman: 0.8364███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▊               | 44/47 [00:00<00:00, 95.09it/s]
Euclidean-Distance:     Pearson: 0.8323 Spearman: 0.8360
Dot-Product-Similarity: Pearson: 0.8000 Spearman: 0.7965


### SimCSE

SimCSE (Gao et al., 2021) also views the identical sentences as the positive examples. The main difference is that SimCSE samples different dropout masks for the same sentence to generate a embedding-level positive pair and uses in-batch negatives. Thus, this learning objective is equivalent to feeding each batch of sentences to the
shared encoder twice and applying the MNRL-loss.

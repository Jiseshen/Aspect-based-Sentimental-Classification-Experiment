# Aspect-based Sentimental Classification Experiment
The task is to predict the sentimental polarity attached to each aspect of a sentence, as SemEval 2014 Task 4 brought about.

Reference:
[SemEval-2014 Task 4: Aspect Based Sentiment Analysis](https://aclanthology.org/S14-2004) (Pontiki et al., SemEval 2014)

Our experiments include the following methods:

## Test of LLMs with Specialized Masking Augmentation
With auxiliary sentences, we can simply fine-tune BERT-like LLMs to work as normal sentiment classification. We proposed specialized masking augmentation which randomly masks the aspect word in case of trivial memorization of fixed patterns, adding to the robustness of convergence, and conducted extensive experiments on several different pre-trained models to show that.

Reference:
[Utilizing BERT for Aspect-Based Sentiment Analysis via Constructing Auxiliary Sentence](https://aclanthology.org/N19-1035) (Sun et al., NAACL 2019)

## Specialized models or pipelines
We implemented Dual-MRC and Syntax-based GAT model, which were specifically designed on the task. We introduced some minor improvements, such as the heuristic of extracted aspect terms.

Reference:
Mao, Y., Shen, Y., Yu, C., & Cai, L. (2021). A Joint Training Dual-MRC Framework for Aspect Based Sentiment Analysis. *ArXiv, abs/2101.00816.*

[Relational Graph Attention Network for Aspect-based Sentiment Analysis](https://aclanthology.org/2020.acl-main.295) (Wang et al., ACL 2020)

## Instruction Learning
We reformatted the training set as examplars prompt, used clustering methods to generate the few-shot examples and tested GPT-3.5 and GPT-4's performance.

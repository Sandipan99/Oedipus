# Oedipus
Code and data related to paper "Evaluating LLMs’ (In)ability to Follow Prompts in QA Tasks"

> Evaluating LLMs’ (In)ability to Follow Prompts in QA Tasks. Aparup Khatua, Tobias Kalmbach, Prasenjit Mitra and Sandipan Sikdar

***Please cite our paper in any published work that uses any of these resources.***
~~~
Coming soon
~~~

## Abstract

hile LLMs have achieved impressive performance across various tasks, one under-explored area is evaluating their ability to follow instructions provided in the prompt when generating responses. In the context of question-answering (QA) tasks, a crucial research gap is *whether LLMs prioritize their own parametric knowledge or the context provided in the prompt when generating an answer*. Ignoring prompts, even when explicitly instructed to follow them, may adversely affect performance and potentially lead to unintended consequences. Additionally, LLMs should be self-reflective (i.e., LLMs should recognize when their knowledge is inadequate) and avoid hallucinations in such scenarios. To address our research question, we propose *Oedipus*, an evaluation framework to evaluate LLMs' ability to follow prompts. We further note that such abilities could also be influenced by contamination (i.e., exposure to datasets during training) and parametric knowledge. Consequently, we develop a novel QA dataset with four types of contexts—*correct*, *masked*, *noisy*, and *absurd contexts* with *recent questions* that LLMs are unlikely to have encountered in pre-training data or corpus and cannot be answered from parametric knowledge. We evaluate eight LLMs through our proposed evaluation framework and observe that LLMs often fail to follow instructions correctly and are not self-reflective.

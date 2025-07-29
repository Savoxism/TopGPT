# TopGPT 
This repository aims to explore the whole finetuning pipeline to model a specific person's style of communication. In this case I chose Andrew Tate due to his unique, blunt speaking styles. Also, it is a good chance for me to practice crawling data from the Internet and generating synthetic data with both open-source and proprietary Large Language Models. 

This finetuning pipeline has 4 stages: Pre-training, continued pre-training, post-training (which includes supervised finetuning and reinforcement learning) and prompt-based learning. It is advisable to utilize use prompts to maximize the model's capabilities before considering further finetuning.

For evaluation pipeline, we will use the cross-entropy loss on the test set, though personally I do not think it is a good metric. We will also evaluate the model's output with human judgment and AI judgement, in hopes of providing a much more accurate evaluation.

Firstly, due to the astronomical cost to pre-train a LLM from scratch, I will begin to pre-train a small Transformer layer as a proof-of-concept. Although the optimization objective is basically the same, training a small language model means we do not have to worry about hardware optimization such as distributed training, parallel training,... When I have resource I will definitely cover these up.

Secondly, we will conduct continued pre-training with data crawled from Youtube. Specifically, I have prepared 350 long-form podcasts from Andrew Tate to try make the model imitate the style and tone of speaking. If only 350 videos prove to be not enough, I will come up with other ways to obtain data such as back-translation and prompt-based techniques.

Thirdly, and probably easier, is supervised fine-tuning. I will try to prepare at least 20k question-answer pairs obtained from knowledge distillation (basically the cool name for having a more powerful LLM generate data for you). It has been empirically proven that the quality of data in this stage is crucial so I probably need some ways to verify the data, preferably with SCORES.

Finally, and probably the most technically demanding aspect, is to further render the model powerful with reinforcement learning. I am considering between Direct Preference Optimization (DPO) or Contrastive Preference Optimization (CPO). Haven't got into this part yet so there's little to cover


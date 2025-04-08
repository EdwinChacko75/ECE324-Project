# Reasonix: Enhancing Base LLM Reasoning with CoT, SFT, and RLHF

This repository contains the code for our ECE324 project, a framework that improves the reasoning performance of LLMs on math tasks through a multi-stage pipeline involving Chain-of-Thought prompting, Reasoning Distillation, Supervised Fine-Tuning (SFT), and Reinforcement Learning with Human Feedback (RLHF).


## Task Description

While many LLM models excel at next-token prediction, but they often fail to follow logical steps required in math tasks.

The objective our project is to **improve LLMs' step-by-step reasoning ability**, enabling these models to:
- Generate consistent and structured thought processes
- Reduce logical errors
- Improve performance on GSM8K benchmark dataset


## Dataset Description

We use **GSM8K**, a benchmark dataset containing 8,800 grade school math problems.

- 7,500 training samples
- 1,300 test samples
- Each sample includes a word problem and a final numerical answer.
- We augment the training dataset using LLaMA 3.2 3B-Instruct to produce COT-structured step-by-step answer for each question.

For RLHF, we also use:
- **PRM800k**: a human-annotated dataset labeling individual reasoning steps as good, okay, or bad.


## Reasonix Development Pipeline

Our pipeline consists of the following key components:
1. **Few-shot Prompting**: Teach the model how to reason through examples.
2. **Reasoning Distillation**: Use a larger model to generate COT-structured training data.
3. **CoT Supervised Fine-Tuning (SFT)**: Fine-tune the baseline model using data of step-by-step solutions generated in the reasoning distillation step.
4. **RLHF with PPO**: Further align model outputs with human-preferred reasoning using Proximal Policy Optimization.

![Pipeline](./assets/pipeline.jpg)


## Evaluation Metrics

We evaluate the models using:

1. **Accuracy** on GSM8K: % of final answers matching ground truth.
2. **Quality of Reasoning in Four Categories**:
   - Category 1: Fully correct reasoning and correct final answer  
   - Category 2: Correct reasoning and arithmetic error  
   - Category 3: Partial reasoning and wrong answer  
   - Category 4: Completely failed reasoning


## Results

### Baseline Performance

![Pie Chart](./assets/piechart.jpg)


### Comparison between baseline and finetuned model

![Bar Chart](./assets/barchart.png)

- **Baseline**: 16.8% accuracy and GSM8K, and 31% responses fell into Category 4, meaning the model completely failed to reason
- **After finetuning**: accuracy improved to **53.6%** on GSM8K, and the number of responses in Category 4 reduced to **18%** — finetuned model are more capable to reason now

### Sample Output Comparison

![Improved Reasoning](./assets/improved_example.jpg)

Top: Baseline model output (repeating steps without proceeding)  
Bottom: Fintuned model (structured CoT, correct answer)


## How to Run Code

_Coming soon... (code setup, training, inference instructions)_


## License

MIT License. See [LICENSE](./LICENSE) for details.

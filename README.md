# Using 'flip-flop neurons' to model the basal ganglia

# Introduction

Understanding and processing sequential data is a fundamental challenge in various domains, including natural language processing, computational neuroscience, and reinforcement learning. Traditional models often struggle with balancing memory efficiency, computational complexity, and performance when dealing with long sequences or highly dynamic tasks. This project explores innovative neural architectures to address these challenges, combining insights from neuroscience and machine learning.

Flip-flop neurons, as introduced in the literature[1], offer a unique approach to sequence processing by leveraging memory-efficient and context-sensitive mechanisms. These neurons have shown promise in solving complex sequence processing tasks with significantly reduced computational overhead compared to traditional recurrent architectures. However, their potential has yet to be fully explored in real-world applications like sentiment analysis and cognitive modeling.

In parallel, hierarchical neural networks (HRNNs)[3] provide a structured approach to sequence processing by breaking down sequences into manageable chunks and progressively integrating their representations. This chunking mechanism mimics how the human brain processes information hierarchically, making HRNNs a compelling addition to flip-flop neuron models.

The basal ganglia, a central structure in the brain responsible for decision-making, action selection, and working memory, offers another avenue for applying these novel architectures. By integrating flip-flop neurons into a basal ganglia-inspired model, this project aims to explore the biological plausibility and computational advantages of these neurons in tasks involving sequential decision-making.

This report outlines the implementation, integration, and evaluation of flip-flop neurons and HRNNs in both sentiment analysis and basal ganglia-inspired models, with the goal of advancing our understanding of efficient sequence processing and working memory functions.

## Objectives

The primary objectives of this project are as follows:

- Implement a bidirectional and fully connected layer of flip-flop neurons for sentiment analysis, as described in the paper “The Flip-Flop Neuron: A Memory Efficient Alternative for Solving Challenging Sequence Processing and Decision-Making Problems”[1].
- Add sequence chunking, in the form of hierarchical neural networks (HRNNs) to the above model and observe its effect on model performance.
- Implement a model of the basal ganglia using the same flip-flop neurons(from the same paper[1]), as described in the paper “A Basal Ganglia Model for Understanding Working Memory Functions in Healthy and Parkinson’s Conditions”[2].
- Add sequence chunking to the flip-flop neurons used in this model, in the form of hierarchical neural networks (HRNNs) to the above model and observe its effect on model performance.

## Literature Review

The integration of flip-flop neurons into neural network architectures has been explored to address challenges in sequential decision-making tasks, particularly those involving long delays. Holla and Chakravarthy [4] introduced networks of flip-flop neurons capable of storing state information over extended periods, thereby facilitating decision-making processes that require the retention and recall of salient stimuli despite intervening delays.

Expanding on this concept, Yoder [5] proposed neural flip-flops (NFFs) as fundamental components in sequential logic systems. These NFFs are designed with minimal neuronal properties of excitation and inhibition, enabling them to replicate known phenomena associated with short-term memory. The explicit and dynamic nature of these networks allows for the generation and testing of predictions related to memory formation, retention, retrieval, and associated errors.

In the realm of computational neuroscience, the basal ganglia have been extensively modeled to understand their role in working memory and decision-making. Chakravarthy et al. [6] provided a comprehensive modeling perspective on the functions of the basal ganglia, elucidating their involvement in action selection and reinforcement learning. Similarly, Gillies and Arbuthnott [7] developed computational models to simulate the basal ganglia's circuitry and its implications for movement disorders. Berns and Sejnowski[8] proposed a model demonstrating how the basal ganglia produce sequences, offering insights into the neural mechanisms underlying sequential behaviors.

Collectively, these studies contribute to a deeper understanding of how specialized neuronal architectures, such as flip-flop neurons, can be integrated into models of the basal ganglia to enhance our comprehension of working memory functions and decision-making processes in both healthy and pathological conditions.

# Methods

## JK Flip-Flop Neuron

### Implementation of base model

The flip-flop neuron works on the principle of the “JK flip-flop” device in digital logic; the cell keeps the previous part of the sequence in mind while processing the next part. This is useful in sequence processing tasks such as sentiment analysis. The given paper[1] proposes a fully connected bidirectional network of such neurons. The input sequence is processed in both directions, thus allowing the neurons to learn a lot of information about the relationship of a part of the input with both the previous and the next part of the input.

In this project, we implemented the described architecture in PyTorch. This is a diagram of the flip-flop architecture[1]:
![image](https://github.com/user-attachments/assets/51cabfb3-ccae-466f-bf1c-65aeb67ca793)


And this is a detailed diagram of the architecture:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/b57fb76f-7bbf-4566-b267-ad7e098c7abe/image.png)

### Addition of memory chunking

Recurrent neural networks, which are commonly used for sequence processing, have historically struggled with remembering long-range dependencies in sequences. A solution to this problem (which is applicable when the input sequence has a hierarchical structure) is to introduce processing at multiple layers in the RNN, where each layer processes the input at 1 level of hierarchy. This type of RNN is called “Hierarchical RNN” or HRNN. The idea of the HRNN was first introduced by S Hihi and Y Bengio[3]. Furthermore, in the flip-flop paper[1],  memory chunking was mentioned as a possible improvement to the existing flip-flop architecture in the last section.

In this project, we integrated HRNNs into the existing architecture by replacing the bidirectional and fully connected layer of flip-flop neurons with a HRNN of flip-flop neurons which divides the input sequence into chunks, processes each chunk to get an intermediate representation and then processes the intermediate representation again. The architecture is shown below:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/6f9adf98-494e-4768-a7d2-6ce7b70bd816/image.png)

## Basal Ganglia

### Implementation of base model

The basal ganglia is a group of nuclei in the brain that controls voluntary movements, emotions, and other behaviors. The paper[2] proposes the following architecture to model a basal ganglia for working memory tasks. 

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/ecb196f1-4adb-46af-9261-d9e761df9a90/image.png)

We implemented this in PyTorch using linear layers for the GPe - STN and GPI output layers. Additionally, we also integrated the RL framework proposed in the paper for calculating the Q-value at time t to minimize the TD error. The equations for TD error and Q-value respectively are as follows:

$$
\delta^{GPi}(j, t) = rew(t) + \gamma Qval_{t+1}(s, a) - Qval(s, a)
$$

$$
Qval_{t+1}(s, a) = Qval(s, a) + \eta \left( rew(t) + \gamma Qval_{t+1}(s, a) - Qval(s, a) \right)
$$

### Addition of memory chunking

In the modified architecture, we replaced the FF neurons in both the striatal d1 and d2 layers with HRNNs of flip-flop neurons. The rest of the architecture was left as is. This is a description of the modified architecture:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/074834cf-21a3-44b4-99cf-e81b6a0ab9a4/image.png)

# Results

## JK Flip-Flop Neuron

Both the Bi-FFNN architecture and the HRNN architecture were trained on the sentiment analysis task. The IMDB movie reviews dataset from kaggle was used, with a 70-20-10 train-val-test split. Both models were trained for 10 epochs with the Adam optimizer and a learning rate of 0.001.

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/7aed677c-a872-4d4a-8846-ccc7151e906d/image.png)

| Model | Validation Loss | Validation Accuracy | Test Loss | Test Accuracy |
| --- | --- | --- | --- | --- |
| Flip-Flop | 0.5673 | 0.8266 | 0.5365 | 0.8216 |
| Flip-Flop + HRNN | 0.5700 | 0.8701 | 0.5900 | 0.8666 |

## Basal Ganglia

Both models were trained on the 1AX2BY working memory task. In this task, the model is trained to recognize sequences that contain the subsequences “1AX” or “2BY”. 10000 sequences of length 25 were used, 8000 for training and 2000 for testing. The dataset for this task was synthetically generated in the model code.

2 Adam optimizers were used one for updating the neural network parameters (learning rate=0.001) and one for updating the RL parameters (such as Q-value table) (learning rate=0.0005). Both models were trained for 50 epochs. Additionally, the following parameters were used:

```python
input_dim = 8 #possible characters: 1, A, X, 2, B, Y, C, D
striatum_units = 20
chunk_units = 20 #used only in HRNN model
hidden_units = 2 * chunk_units #used only in HRNN model
chunk_size = 5 #used only in HRNN model
stn_gpe_units = 20
```

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/ab5260bc-acb6-4472-9ccb-e137c4b97ad9/image.png)

| Model | Final Loss | Final Accuracy |
| --- | --- | --- |
| Basal Ganglia with HRNN | 0.0066 | 99.85% |
| Base Basal Ganglia | 0.0113 | 99.85% |

### Results of blocking one striatum pathway at a time for both BG models

Furthermore, we observed the effect of allowing only one striatal pathway (d1 or d2) to remain open at a time. The results are as follows:

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/9c953745-d819-4370-b3db-99ca05fb899f/image.png)

![image.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/429de907-7607-47c5-9da4-130d3a5fdb00/f6c6ada5-ce29-4f69-9ea4-e6fc6c527206/image.png)

# Discussion

## JK Flip-Flop Neuron

The model with the HRNN performed better than the base model on the sentiment analysis task. This suggests that the HRNN was able to enhance the sequence-processing capability of the existing flip-flop network by improving the model’s ability to remember long-range dependencies.

## Basal Ganglia

Although both the base model and the model with the HRNN had the same final accuracy, the HRNN model converged to a high accuracy and low loss almost instantly, whereas the base model took about 15 epochs to converge to a similar high accuracy and low loss.

This is likely because of a few reasons. The HRNN architecture processes sequences in smaller chunks and then combines the outputs for higher-level processing. This hierarchical approach enables the model to focus on meaningful sub-segments of data before combining them for a global understanding. This reduces the effective length of dependencies the model needs to learn at each step, simplifying the learning task and accelerating convergence. Additionally, the chunk-based processing allows gradients to propagate through shorter paths within each chunk. This mitigates the vanishing gradient problem often encountered in long sequences, leading to more effective weight updates (and hence, still faster convergence).

We also analyzed the effect of blocking one striatal pathway at a time for both the base and HRNN models of the basal ganglia. For the base model, blocking the d1 pathway only delayed the convergence of the model to a high accuracy and low loss, to epoch 35, whereas blocking the d2 pathway caused the model to get stuck at 48-50% accuracy with next to no improvements in the loss, over 50 epochs. This could be because blocking the d2 pathway also means blocking the STN-GPe layer, which is an integral part of the model; missing out on the additional processing by this layer is likely what prevents the model from converging to a high accuracy. However, the same result was not observed for the BG model with HRNNs; the model converged to a high accuracy and low loss instantly irrespective of which pathway was blocked. This suggests that the additional processing of the HRNN makes up for the deficit caused by the absence of the STN-GPe layer.

# References

[1]:  Sweta Kumari, Vigneswaran Chandrasekaran, and V Srinivasa Chakravarthy. The flip-flop neuron: a memory efficient alternative for solving challenging sequence processing and decision-making problems. *Neural Computing and Applications*, 35(34):24543–24559, 2023.

[2]: C Vigneswaran, Sandeep Sathyanandan Nair, and V Srinivasa Chakravarthy. A basal ganglia model for understanding working memory functions in healthy and parkinson’s conditions. *Cognitive Neurodynamics*, pages 1–17, 2024.

[3]: Hihi, Salah, and Yoshua Bengio. "Hierarchical recurrent neural networks for long-term dependencies." *Advances in neural information processing systems* 8 (1995).

[4]: P. Holla and S. Chakravarthy, "Decision making with long delays using networks of flip-flop neurons," 2016 International Joint Conference on Neural Networks (IJCNN), Vancouver, BC, Canada, 2016, pp. 2767-2773, doi: 10.1109/IJCNN.2016.7727548. keywords: {Neurons;Biological neural networks;Training;Standards;Mathematical model;Backpropagation;Flip-flops}

[5]: Yoder L (2024) Neural flip-flops I: Short-term memory. PLoS ONE 19(3): e0300534. https://doi.org/10.1371/journal.pone.0300534 

[6]: Chakravarthy, V. Srinivasa, Denny Joseph, and Raju S. Bapi. "What do the basal ganglia do? A modeling perspective." *Biological cybernetics* 103 (2010): 237-253.

[7]: Gillies, Andrew, and Gordon Arbuthnott. "Computational models of the basal ganglia." *Movement disorders* 15.5 (2000): 762-770.

[8]: Berns, Gregory S., and Terrence J. Sejnowski. "A computational model of how the basal ganglia produce sequences." *Journal of cognitive neuroscience* 10.1 (1998): 108-121.

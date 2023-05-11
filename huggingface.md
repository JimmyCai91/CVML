**Text Classification**
1. What is the standard fine-tuning approach discussed in this chapter

---

**Transformers Anatomy**
1. Name the most important `building block` that makes a transformer work (p60)
1. There are several ways to implement a `answer to the 
  question above`, but the most common one is ____, 
  from the paper introducing the Transformer architecture. 
  `List its implementation steps` and then `draw its network` (p62-63 \& Fig. 3-4)
1. What network improves the mentioned approach? (page 67) Draw its network (Fig. 3-5)
1. How does transformers differ from recurrent models (page 61)
1. Although there are hundreds of different transformer models, most of them 
  belong to one of three types. `Name them, list their representative models,
  and provide their applications` (page 59, page 79-83)
1. Describe the feed-forward layer in transformers and explain its 
  main hyperparameter (page 70-71)
1. Explain/compare post layer normalization and pre layer normalization in 
  transformers (page 71-72)


<font color=#ff0000>nn.GELU()</font>

    `nn.GELU` is a class in PyTorch that applies the **Gaussian Error Linear Units function**¹. The function is defined as `GELU(x) = x * Φ(x)`, where `Φ(x)` is the Cumulative Distribution Function for Gaussian Distribution¹.

    The `approximate` argument can be set to `'tanh'` to estimate the Gelu function with `GELU(x) = 0.5 * x * (1 + Tanh(√(2/π) * (x + 0.044715 * x^3)))`¹.

    Is there anything else you would like to know about this topic?

    Source: Conversation with Bing, 5/10/2023
    (1) GELU — PyTorch 2.0 documentation. https://pytorch.org/docs/stable/generated/torch.nn.GELU.html.
    (2) Gaussian error linear unit (GELU) layer - MATLAB. https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.gelulayer.html.
    (3) arXiv:1606.08415v4 [cs.LG] 8 Jul 2020. https://arxiv.org/pdf/1606.08415.pdf.
    (4) Gaussian Error Linear Unit (GELU) - OpenGenus IQ: Computing Expertise .... https://iq.opengenus.org/gaussian-error-linear-unit/.
    (5) tensorflow.nn.gelu Example. https://programtalk.com/python-more-examples/tensorflow.nn.gelu/.

    A **Gaussian error linear unit (GELU) layer** weights the input by its probability under a Gaussian distribution¹. This operation is given by `GELU(x) = x * Φ(x)`, where `Φ(x)` is the standard Gaussian cumulative distribution function².

    The GELU nonlinearity weights inputs by their value, rather than gating inputs by their sign as in ReLUs². It has been found to improve performance across computer vision, natural language processing, and speech tasks².

    Is there anything else you would like to know about this topic?

    Source: Conversation with Bing, 5/10/2023
    (1) Gaussian error linear unit (GELU) layer - MATLAB. https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.gelulayer.html.
    (2) [1606.08415] Gaussian Error Linear Units (GELUs) - arXiv. https://arxiv.org/abs/1606.08415.
    (3) GELU Explained | Papers With Code. https://paperswithcode.com/method/gelu.
    (4) Gaussian error linear unit (GELU) layer - MATLAB - MathWorks 日本. https://jp.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.gelulayer.html.
    (5) Gaussian error linear unit (GELU) layer - MATLAB - MathWorks América Latina. https://la.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.gelulayer.html.

    One advantage of using a **Gaussian error linear unit (GELU) layer** is that it has been found to improve performance across computer vision, natural language processing, and speech tasks when compared to other activation functions such as ReLU and ELU¹⁴.

    The GELU nonlinearity weights inputs by their value, rather than gating inputs by their sign as in ReLUs¹. This can be considered a smoother version of ReLU².

    I couldn't find any specific disadvantages of using GELU in my search results. Is there anything else you would like to know about this topic?

    Source: Conversation with Bing, 5/10/2023
    (1) [1606.08415] Gaussian Error Linear Units (GELUs) - arXiv. https://arxiv.org/abs/1606.08415.
    (2) Gaussian Error Linear Unit Activates Neural Networks Beyond ReLU. https://syncedreview.com/2020/01/03/gaussian-error-linear-unit-activates-neural-networks-beyond-relu/.
    (3) GELU Explained | Papers With Code. https://paperswithcode.com/method/gelu.
    (4) arXiv:1606.08415v2 [cs.LG] 8 Jul 2016. https://arxiv.org/pdf/1606.08415v2.pdf.

---

**Multilingual Named Entity Recognition**

---

**Making Transformers Efficient in Production**
1. name the four complementary techniques that can be used to speed up the 
  predictions and reduce the memory footprint of transformers (page 209)
2. name the main approaches to quantization and list their pros and cons
  (page 235)

---

**Dealing with Few to No Labels**
1. Draw `Figure 9-1` on page 250, which shows several techniques that can be 
   used to improve model performance in the absence of large amounts of 
   labeled data
2. If you find it difficult to get good results on your own dataset, here are 
   few things you can do to improve the zero-shot pipeline: A \& B (page 271)
3. In NLP practice, name the two types of data augmentation techniques that are commonly used (page 272)
4. List steps of using embeddings as a lookup table (page 275)
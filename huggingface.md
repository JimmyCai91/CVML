**Transformers Anatomy**
- Name the most important building block that makes a transformer work 
- There are several ways to implement a <font color=#ff0000>answer to the 
  question above</font>, but the most common one is ____, 
  from the paper introducing the Transformer architecture. <font color=#ff0000>
  List its implementation steps</font>
- How does transformers differ from recurrent models
- Although there are hundreds of different transformer models, most of them 
  belong to one of three types. <font color=#ff0000>Name them and provide their
  applications</font> (page 59)
- Describe the feed-forward layer in transformers and explain its 
  main hyperparameter
- Explain/compare post layer normalization and pre layer normalization in 
  transformers
- nn.GELU()  
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
- name the four complementary techniques that can be used to speed up the 
  predictions and reduce the memory footprint of transformers (page 209)
- name the main approaches to quantization and list their pros and cons
  (page 235)
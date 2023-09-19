### 总述

本工作为 [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 的复赛题目，即自选大型语言模型，并使用 TensorRT 或 TensorRT-LLM 进行模型推理优化，我们队伍采用TensorRT-LLM完成对新模型[DeciCoder](https://huggingface.co/Deci/DeciCoder-1b)的构建以及推理优化。

[DeciCoder](https://huggingface.co/Deci/DeciCoder-1b)由深度学习公司Deci推出，可以用各种编程语言生成代码。该模型Decoder layer采用 Grouped Query Attention，其Position Embeddings采用Rotary Position Embeddings， 拥有 1.1B个参数和包含 2048 个标记的大型上下文窗口，能够生成高质量且多样化的代码片段。其huggingface地址为：[https://huggingface.co/Deci/DeciCoder-1b](https://huggingface.co/Deci/DeciCoder-1b)。

模型搭建完的效果：

代码运行步骤见/tensorrt_llm_july-release-v1/examples/deci下的README.md

### 主要开发工作

#### 开发工作的难点

**DeciCoder**模型的Decode layer采用了Grouped Query Attention以及其Position Embeddings采用Rotary Position Embeddings，由于该版本的TensorRT-LLM的Rotary Position Embeddings包含在gpt attention plugin中，因此该模型Rotary Position Embeddings需要采用TensorRT-LLM中的function去进行实现，并构建Grouped Query Attention，再根据torch版本的模型构建TensorRT-LLM中的DeciCoder模型。

工程实现中由于TensorRT-LLM构建的为静态图，而在torch版本中的DeciCoder构建时，其attention的forward会根据是否存在past_kv_value判断flag标志，根据flag标志判断是否采用mask，其属于一个动态图。因此采用TensorRT-LLM构建DeciCoder的Grouped Query Attention时需要重新处理该逻辑。DeciCoder中的attention在第一次时会使用mask操作外，其余过程中均不使用mask操作。我们队伍将该逻辑的实现放到了TensorRT-LLM的generation中进行实现。debug中的难点：


### 开发与优化过程

**DeciCoder**模型的构建流程：

- 阅读TensorRT-LLM下example的代码，理解TensorRT-LLM的运作流程。
- 根据huggingface中提供的modeling_decicoder.py将**DeciCoder**模型的构建流程理清。
- 使用TensorRT-LLM的functional.py中的function进行模型的构建
  + 实现Grouped Query Attention的repeat_kv操作
  + 实现Rotary Position Embeddings
  + 实现Grouped Query Attention，重新处理其逻辑，符合静态图的要求
  + 搭建DeciCoder model以及casual model
- 使用TensorRT-LLM的test api对上述模型构建部分进行测试，保证其输出维度的正确性。
- 编写bulid.py构建engine
- 编写generation.py和run.py得到运行结果

**DeciCoder**模型的特别步骤：
+ 1
+ 2

### 优化效果

+ 精度：由于**DeciCoder**模型用于生成代码，其summarize.py评判的机制不太符合生成代码的要求，因此精度的测试我们采用TensorRT-LLM构建模型结果与huggingface模型结果进行对比

如下图：


+ 性能：

pass

### Bug报告（可选）



### 送分题答案（可选）

送分题1：

![](tensorrt_llm_july-release-v1/examples/deci/songfenti1.png)

送分题2：

![](tensorrt_llm_july-release-v1/examples/deci/songfenti2.png)

### 经验与体会（可选）



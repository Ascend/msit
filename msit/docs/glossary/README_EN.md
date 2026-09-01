# Glossary

## om File

- Offline inference file on the NPU. Similar to onnx files and pb files of the TensorFlow framework.

## atb (Ascend Transformer Boost)

- Ascend Transformer Boost is an inference acceleration engine library for Transformer-based neural networks. The library contains highly optimized modules for various Transformer models, such as the Encoder and Decoder parts. As an acceleration library for Transformer models (Ascend Transformer Boost), it improves Transformer model performance and provides basic high-performance operators and efficient operator combination technology (Graph) for convenient model acceleration. It can be used by various model inference frameworks. Currently, users include PyTorch, MindSpore, and Paddle.
- For details, refer to the [corresponding chapter in the Development Guide of the Ascend Community CANN Development Kit](https://www.hiascend.com/document/detail/zh/canncommercial/700/foundmodeldev/ascendtb/).

## torchair (torch Graph Mode)

* torchair provides users with an efficient and flexible model deployment solution, making it easier for users to apply models in real scenarios. torchair converts the FX graph of torch into a GE computation graph and provides compilation and execution interfaces for the GE computation graph. The FX graph is an intermediate representation in PyTorch, used to represent the computation graph and operation sequence of the model. The GE computation graph is the computation graph of the Ascend AI processor, used to represent the computation graph and operation sequence of the model. Converting the FX graph into a GE computation graph enables cross-platform model deployment and accelerates model inference.
* For details, refer to the [corresponding chapter in the Development Guide of the Ascend Community CANN Development Kit](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/torchairuseguide/torchair_0002.html)

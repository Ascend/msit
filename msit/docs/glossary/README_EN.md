# Term List

## OM file

 * Offline inference file on the NPU. Similar to the onnx file, the pb file of the TensorFlow framework

## atb (Ascend Transformer Boost)

 * Transformer inference acceleration library (Ascend Transformer Boost) is a neural network inference acceleration engine library based on Transformer. The library contains highly optimized modules of various Transformer models, such as Encoder and Decoder. Transformer model-oriented acceleration library (Ascend Transformer Boost) improves the performance of Transformer models and provides basic high-performance operators and efficient operator combination techniques (Graph) to facilitate model acceleration. Various model inference frameworks are available, including PyTorch, MindSpore, and Paddle.
 * For details, see.[Chapters in the Ascend Community CANN Development Kit](https://www.hiascend.com/document/detail/zh/canncommercial/700/foundmodeldev/ascendtb/)    

## torchair (torch pattern mode)

 * Torchair provides an efficient and flexible model deployment solution, which enables users to easily apply models to actual scenarios. Torchair converts the torch FX diagram into a GE calculation diagram, and provides the compilation and execution interfaces for the GE calculation diagram. FX diagram is an intermediate representation in PyTorch, used to represent the computation diagram and operation sequence of a model. The GE calculation diagram is the calculation diagram of the Ascend AI processor and is used to represent the calculation diagram and operation sequence of the model. Converting FX graphs into GE compute graphs can realize cross-platform model deployment and accelerate model inference.
 * For details, see.[Corresponding chapters in the development guide in the Ascend Community CANN Development Kit](https://www.hiascend.com/document/detail/zh/Pytorch/700/modthirdparty/torchairuseguide/torchair_0002.html)    

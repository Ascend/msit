# List Command #

## Introduction ##

Lists all knowledge bases that support automatic optimization.

## Running an Example ##

```
msit debug surgeon list
```

An example output is as follows:

```
Available knowledges:
   0 KnowledgeConv1d2Conv2d
   1 KnowledgeMergeConsecutiveSlice
   2 KnowledgeTransposeLargeInputConv
   3 KnowledgeMergeConsecutiveConcat
   4 KnowledgeTypeCast
   5 KnowledgeSplitQKVMatmul
   6 KnowledgeSplitLargeKernelConv
   7 KnowledgeResizeModeToNearest
   8 KnowledgeTopkFix
   9 KnowledgeMergeCasts
  10 KnowledgeEmptySliceFix 
  11 KnowledgeDynamicReshape
  12 KnowledgeGatherToSplit
  13 KnowledgeAvgPoolSplit
  14 KnowledgeBNFolding
  15 KnowledgeModifyReflectionPad
  16 KnowledgeBigKernel
```

The listed knowledge base is displayed in the format of SN+Name. When the evaluate or optimize command is used to specify a knowledge base, you can specify the SN or name of the knowledge base. For details about specific knowledge bases, see[Knowledge Base Document](../../../../../components/debug/surgeon/docs/knowledge_optimizer/knowledge_optimizer_rules.md)	.

Note: The sequence number exists for manual invoking. The sequence number may change because the knowledge base may be deleted or modified.


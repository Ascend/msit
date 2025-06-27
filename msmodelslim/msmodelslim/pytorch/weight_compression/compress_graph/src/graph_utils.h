/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: graph_utils.h for weight compression
 * Author: Huawei
 * Create: 2023-09-21
 */

#ifndef DAVINCI_GRAPH_UTILS_H
#define DAVINCI_GRAPH_UTILS_H
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <iostream>

#include "graph.h"
#include "types.h"
#include "tensor.h"
#include "ge_error_codes.h"
#include "ge_api_types.h"
#include "ge_api.h"
#include "all_ops.h"

#define FAILED (-1)
#define SUCCESS 0

using namespace ge;
using std::vector;
using std::string;

namespace GraphUtils {
int CheckShape(vector<int64_t> &shape);

void GetDataSizeFromShape(vector<int64_t> shape, int64_t &size);

bool GetDataFromBin(string input_path, vector<int64_t> shapes, uint8_t** data, int data_type_size);

int32_t BuildCompressFcGraph(Graph &graph, uint8_t* data, vector<int64_t> &shape, vector<int64_t> &compressParameters);
}
#endif // DAVINCI_GRAPH_UTILS_H

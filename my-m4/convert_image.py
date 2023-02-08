# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

#-*- coding : utf-8-*-
# coding:unicode_escape
import os
import pathlib
import sys
import numpy as np


def create_header_file(name, tensor_name, tensor_data, output_path):
    """
    This function generates a header file containing the data from the numpy array provided.
    """
    file_path = pathlib.Path(f"{output_path}/" + name).resolve()
    # Create header file with npy_data as a C array
    raw_path = file_path.with_suffix(".h").resolve()
    # tensor_data = tensor_data[0]
    with open(raw_path, "w") as header_file:
        header_file.write(
            "\n"
            + f"const size_t {tensor_name}_len = {tensor_data.size};\n"
            + f'__attribute__((section(".data.tvm"), aligned(16))) float {tensor_name}[] = '
            +"{"
        )
        # for i in range(1):
            # header_file.write("{")
        for num in tensor_data:
            header_file.write(f'{num},')
            # header_file.write("},")
        header_file.write("};\n\n")


def create_headers(image_name):
    """
    This function generates C header files for the input and output arrays required to run inferences
    """
    img_path = os.path.join("./", f"{image_name}")

    '''
    CWRU dataset is a set of 1D data
    '''
    img_data = []
    with open(image_name, "r") as f:
        # all = f.readlines()
        # for i in range(936):
        #     img_data.append((all[i].split()))
        img_data = f.readline().split()

    img_data = np.asarray(img_data).astype("float32") 
    # # Add the batch dimension, as we are expecting 4-dimensional input: NCHW.
    # img_data = np.expand_dims(img_data, axis=0)
    # img_data = np.expand_dims(img_data, axis=2)
    # print(img_data.shape)
    # Create input header file
    input_data = img_data
    input_data = input_data.astype(np.float32)
    # print(input_data)
    create_header_file("inputs", "input", input_data, "./include")
    # Create output header file
    output_data = np.zeros([10], np.float32)
    create_header_file(
        "outputs",
        "output",
        output_data,
        "./include"
    )


if __name__ == "__main__":
    create_headers(sys.argv[1])

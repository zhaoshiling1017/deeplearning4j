/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
// @author Yurii Shyrma (iuriish@yahoo.com), created on 07.06.2018
//
#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_mirror_pad)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops {


CUSTOM_OP_IMPL(mirror_pad, 2, 1, false, 0, 1) {

    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);
    
    NDArray<T>* output  = OUTPUT_VARIABLE(0);

    const int mode = INT_ARG(0);    // 0 - REFLECT, else - SYMMETRIC
    const int includeBorder = mode ? 0 : 1;

    if(input->rankOf() <= 1) {  // when in put scalar or vector;
        REQUIRE_TRUE(paddings->lengthOf() == 2, 0, "MIRROR_PAD OP: the length of paddings array must be equal 2, when input array is vector or scalar, bot but got %i instead !", paddings->rankOf());
        REQUIRE_TRUE( ((*paddings)(0.) <= (input->lengthOf() - includeBorder)) && ((*paddings)(1) <= (input->lengthOf() - includeBorder)), 0, "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then length of input array (being vector or scalar) for symmetric mode (or length-1 for reflect mode) !");    
    }
    else {
        REQUIRE_TRUE(paddings->rankOf() == 2, 0, "MIRROR_PAD OP: the rank of paddings array must be equal 2, but got %i instead !", paddings->rankOf());
        REQUIRE_TRUE(paddings->sizeAt(0) == input->rankOf(), 0, "MIRROR_PAD OP: zero dimension of paddings array must be equal to input array rank, but got %i and %i correspondingly !", paddings->sizeAt(0), input->rankOf());
        for(int i = 0; i < input->rankOf(); ++i)
            REQUIRE_TRUE( ((*paddings)(i,0) <= (input->sizeAt(i) - includeBorder)) && ((*paddings)(i,1) <= (input->sizeAt(i) - includeBorder)), 0, "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then corresponding dimension of input array for symmetric mode (or dimension-1 for reflect mode) !");    
    }

    helpers::mirrorPad<T>(*input, *paddings, *output, mode);

    return Status::OK();
}
 

DECLARE_SHAPE_FN(mirror_pad) {

    NDArray<T>* input    = INPUT_VARIABLE(0);
    NDArray<T>* paddings = INPUT_VARIABLE(1);

    const int rank = input->rankOf() ? input->rankOf() : 1;                 // if scalar is input then vector is output
    const int includeBorder = static_cast<bool>(INT_ARG(0)) ? 0 : 1;        // 0 - REFLECT, else - SYMMETRIC

    if(rank == 1) {  // when in put scalar or vector;
        REQUIRE_TRUE(paddings->lengthOf() == 2, 0, "MIRROR_PAD OP: the length of paddings array must be equal 2, when input array is vector or scalar, bot but got %i instead !", paddings->rankOf());
        REQUIRE_TRUE( ((*paddings)(0.) <= (input->lengthOf() - includeBorder)) && ((*paddings)(1) <= (input->lengthOf() - includeBorder)), 0, "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then length of input array (being vector or scalar) for symmetric mode (or length-1 for reflect mode) !");
    }
    else {
        REQUIRE_TRUE(paddings->rankOf() == 2, 0, "MIRROR_PAD OP: the rank of paddings array must be equal 2, but got %i instead !", paddings->rankOf());
        REQUIRE_TRUE(paddings->sizeAt(0) == input->rankOf(), 0, "MIRROR_PAD OP: zero dimension of paddings array must be equal to input array rank, but got %i and %i correspondingly !", paddings->sizeAt(0), input->rankOf());
        for(int i = 0; i < input->rankOf(); ++i)
            REQUIRE_TRUE( ((*paddings)(i,0) <= (input->sizeAt(i) - includeBorder)) && ((*paddings)(i,1) <= (input->sizeAt(i) - includeBorder)), 0, "MIRROR_PAD OP: wrong content of paddings array, its elements must be no grater then corresponding dimension of input array for symmetric mode (or dimension-1 for reflect mode) !");    
    }
    
    Nd4jLong* outShapeInfo(nullptr);

    if(rank == 1) {
        Nd4jLong len = input->lengthOf() + paddings->getScalar(0L) + (*paddings)(1);
        outShapeInfo = ShapeUtils<T>::createVectorShapeInfo(len, block.getWorkspace());
    }
    else {
        ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
        outShapeInfo[0] = rank;
        for(int i = 0; i < rank; ++i)
            outShapeInfo[i+1] = input->sizeAt(i) + (*paddings)(i,0) + (*paddings)(i,1);
        shape::updateStrides(outShapeInfo, input->ordering());
    }

    return SHAPELIST(outShapeInfo);
}


}
}

#endif
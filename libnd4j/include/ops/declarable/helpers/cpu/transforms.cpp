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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 20.04.2018
//


#include<ops/declarable/helpers/transforms.h>
#include <array/ResultSet.h>
#include <helpers/ShapeUtils.h>
#include <numeric>

namespace nd4j 	  {
namespace ops 	  {
namespace helpers {



//////////////////////////////////////////////////////////////////////////
template <typename T>
void triu(const NDArray<T>& input, NDArray<T>& output, const int diagonal) {

    const int rank = input.rankOf();
    
    switch(rank) {

        case 1:
            for(int i = 0; i < output.sizeAt(0); ++i)
                output({i, i+1, 0,0}).assign(input);
            output.setValueInDiagMatrix(0., diagonal-1, 'l');    
            break;

        case 2:
            output.assign(input);
            output.setValueInDiagMatrix(0., diagonal-1, 'l');    
            break;

        default: 
            ResultSet<T>* inTads  = input.allTensorsAlongDimension({rank-2, rank-1});
            ResultSet<T>* outTads = output.allTensorsAlongDimension({rank-2, rank-1});                        

// #pragma omp parallel for schedule(guided) if(inTads->size() > Environment::getInstance()->elementwiseThreshold()) 
            for(int i = 0; i < inTads->size(); ++i) {
                NDArray<T>* inSubArr = inTads->at(i);
                NDArray<T>* outSubArr = outTads->at(i);
                outSubArr->assign(inSubArr);
                outSubArr->setValueInDiagMatrix(0., diagonal-1, 'l');
            }
            delete inTads;
            delete outTads;
    }
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void triuBP(const NDArray<T>& input, const NDArray<T>& gradO, NDArray<T>& gradI, const int diagonal) {

    NDArray<T> dOdI(&gradO);                // dO/dI
    helpers::triu(input, dOdI, diagonal);

#pragma omp parallel for if(dOdI.lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
    for(int i = 0; i < dOdI.lengthOf(); ++i) {
        T* currElement = &dOdI(i);
        if(*currElement != (T)0.)
            *currElement = 1.;
    }

    gradI.assign(dOdI * gradO);                          // chain rule: dLoss/dI = dO/dI * dLoss/dO 
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void trace(const NDArray<T>& input, NDArray<T>& output) {

    const int inRank = input.rankOf();

    ResultSet<T>* setOfSubArrs = input.allTensorsAlongDimension({inRank-2, inRank-1});

#pragma omp parallel for if(setOfSubArrs->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
    for(int i = 0; i < setOfSubArrs->size(); ++i)
        output(i) = setOfSubArrs->at(i)->getTrace();

    delete setOfSubArrs;
}


//////////////////////////////////////////////////////////////////////////
template <typename T>
void randomShuffle(NDArray<T>& input, NDArray<T>& output, nd4j::random::RandomBuffer& rng, const bool isInplace) {

    // check edge cases first
    int temp;
    const int firstDim = input.sizeAt(0);    
    if(input.lengthOf() == 1 || firstDim == 1) {
        
        if(!isInplace)
            output.assign(input);
    } 
    else if (input.isVector() || shape::isLikeVector(input.getShapeInfo(), temp)) {
                
        // apply Fisher-Yates shuffle 
        if(isInplace) {
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                if(i == r)
                    continue;
                math::nd4j_swap<T>(input(i), input(r));            
            }        
        }
        else {        
            std::vector<int> indices(firstDim);        
            std::iota(indices.begin(), indices.end(), 0);        
            output(0.) = input(0.);
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                output(i) = input(indices[r]);
                if(i == r)
                    continue;
                output(r) = input(indices[i]);                
                math::nd4j_swap<int>(indices[i], indices[r]);
            }           
            rng.rewindH(firstDim-1);
        }
    }
    else {
            
        // evaluate sub-arrays list of input array through all dimensions excluding first one
        std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input.rankOf(), {0});       
        ResultSet<T>* subArrsListIn = input.allTensorsAlongDimension(dimensions);

        // apply Fisher-Yates shuffle
        if(isInplace) {
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)        
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                if(i == r)
                    continue;
                subArrsListIn->at(i)->swapUnsafe(*subArrsListIn->at(r));
            }        
        }
        else {
            // evaluate sub-arrays list of output array through all dimensions excluding first one        
            ResultSet<T>* subArrsListOut = output.allTensorsAlongDimension(dimensions);        
            std::vector<int> indices(firstDim);        
            std::iota(indices.begin(), indices.end(), 0);        
            bool isZeroShuffled = false;
#pragma omp parallel for if((firstDim-1) > Environment::getInstance()->elementwiseThreshold()) schedule(guided)       
            for(int i = firstDim-1; i > 0; --i) {
                int r = rng.nextInt(0, i);
                subArrsListOut->at(i)->assign(subArrsListIn->at(indices[r]));
                if(r == 0)
                    isZeroShuffled = true;
                if(i == r)
                    continue;
                subArrsListOut->at(r)->assign(subArrsListIn->at(indices[i]));
                math::nd4j_swap<int>(indices[i], indices[r]);
            }           
            if(!isZeroShuffled)
                subArrsListOut->at(0)->assign(subArrsListIn->at(0));
            delete subArrsListOut;
        }
        rng.rewindH(firstDim-1);
        delete subArrsListIn;
    }

}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void pad(const int mode, const NDArray<T>& input, const NDArray<T>& paddings, NDArray<T>& output, const T padValue ) {

    const int rank = output.rankOf();
    std::vector<int> dimsToExclude(rank);
    std::iota(dimsToExclude.begin(), dimsToExclude.end(), 0);             // fill with 0, 1, ... rank-1    

    Nd4jLong numLeft    = paddings(rank-1,0);
    Nd4jLong numRight   = paddings(rank-1,1);
    Nd4jLong inDimSize  = input.sizeAt(rank-1);
    Nd4jLong outDimSize = output.sizeAt(rank-1);

    std::vector<std::vector<Nd4jLong>> outIdx = { std::vector<Nd4jLong>(2*rank), {numLeft, numLeft + inDimSize}, {0, numLeft}, {numLeft + inDimSize, outDimSize} };
    
    for(int i = 0; i < rank-1; ++i) {
        outIdx[0][2*i]     = paddings(i, 0);
        outIdx[0][2*i + 1] = outIdx[0][2*i] + input.sizeAt(i);
    }    
    outIdx[0][2*rank-1] = outIdx[0][2*rank-2] = 0;

    // ***** populate innermost sub-arrays firstly ***** //
    dimsToExclude.pop_back();    

    Nd4jLong startL = mode == 1 ? 1 : 0;                            // REFLECT or SYMMETRIC
    Nd4jLong startR = mode == 1 ? inDimSize-2 : inDimSize-1;        // REFLECT or SYMMETRIC

    Nd4jLong numOfSubArrs = ShapeUtils<T>::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);

    NDArray<T> outSubArr0 = output(outIdx[0], true);
    
#pragma omp parallel for schedule(guided)
    for(Nd4jLong j = 0; j < numOfSubArrs; ++j) {

        NDArray<T> outSubArr1   = outSubArr0(j, dimsToExclude);
        NDArray<T> inSubArr     = input(j, dimsToExclude);        
        NDArray<T> outSubArrMid = outSubArr1(outIdx[1]);

        outSubArrMid.assign(inSubArr);      // assign middle

        if(mode == 0)  { // CONSTANT
            if(numLeft != 0) {
                NDArray<T> temp = outSubArr1(outIdx[2]);
                temp = padValue;                        // assign left                     
            }
            if(numRight != 0) {
                NDArray<T> temp = outSubArr1(outIdx[3]);
                temp = padValue;                        // assign right
            }
        }
        else {                                                              // REFLECT or SYMMETRIC
            
            for(Nd4jLong k = numLeft-1, e = startL; k >= 0; --k, ++e)     // fill left side             
                outSubArr1(k) = inSubArr(e);            

            for(Nd4jLong k = numLeft + inDimSize, e = startR; k < outDimSize; ++k, --e)     // fill right side
                outSubArr1(k) = inSubArr(e);                        
        }
    }        

    // ***** fill rest of outer sub-arrays ***** //    
    std::vector<Nd4jLong> outIdxInner(2,0);
    std::vector<Nd4jLong> outIdxOuter(2,0);

    for(int i = rank - 2; i >= 0; --i) {
        
        dimsToExclude.pop_back();

        outIdxInner.push_back(0), outIdxInner.push_back(0);
        outIdxOuter.push_back(0), outIdxOuter.push_back(0);

        Nd4jLong numLeft  = paddings(i,0);
        Nd4jLong numRight = paddings(i,1);

        if(numLeft == 0 && numRight == 0)
            continue;

        Nd4jLong inDimSize  = input.sizeAt(i);
        Nd4jLong outDimSize = output.sizeAt(i);
        
        if(mode == 0) {
            outIdxOuter[0] = 0;                   outIdxOuter[1] = numLeft;
            outIdxInner[0] = numLeft + inDimSize; outIdxInner[1] = outDimSize;
        }
        
        startL = mode == 1 ? numLeft+1 : numLeft;                            // REFLECT or SYMMETRIC
        startR = mode == 1 ? numLeft+inDimSize-2 : numLeft+inDimSize-1;      // REFLECT or SYMMETRIC
        
        numOfSubArrs = ShapeUtils<T>::getNumOfSubArrs(output.getShapeInfo(), dimsToExclude);

#pragma omp parallel for schedule(guided) firstprivate(outIdxOuter, outIdxInner)
        for(Nd4jLong j = 0; j < numOfSubArrs; ++j) {

            NDArray<T> outSubArr = output(j, dimsToExclude);

            if(mode == 0)  { // CONSTANT

                if(numLeft != 0) {                   
                    NDArray<T> temp = outSubArr(outIdxOuter);
                    temp = padValue;                              // assign left 
                }
        
                if(numRight != 0) {                   
                    NDArray<T> temp = outSubArr(outIdxInner);
                    temp = padValue;                              // assign right
                }
            }
            else {                                                              // REFLECT or SYMMETRIC
            
                for(Nd4jLong k = numLeft-1, e = startL; k >= 0; --k, ++e) {    // fill left side
                    outIdxOuter[0] = k;
                    outIdxOuter[1] = k+1;
                    outIdxInner[0] = e;
                    outIdxInner[1] = e+1;
                    NDArray<T> outSubArrInner = outSubArr(outIdxInner);
                    NDArray<T> outSubArrOuter = outSubArr(outIdxOuter);
                    outSubArrOuter.assign(outSubArrInner);
                }

                for(Nd4jLong k = numLeft + inDimSize, e = startR; k < outDimSize; ++k, --e) {    // fill right side
                    outIdxOuter[0] = k;
                    outIdxOuter[1] = k+1;
                    outIdxInner[0] = e;
                    outIdxInner[1] = e+1;
                    NDArray<T> outSubArrInner = outSubArr(outIdxInner);
                    NDArray<T> outSubArrOuter = outSubArr(outIdxOuter);
                    outSubArrOuter.assign(outSubArrInner);
                }
            }
        }        
    }
}



////////////////////////////////////////////////////////////////////////
/*// initial values of inIdx, outIdx, dim must be equal to zero
template<typename T>
void recursiveLoopForPad(const int mode, NDArray<T>& input, const NDArray<T>& paddings, NDArray<T>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, T padValue ) {
    
    int leftOffset;
    // dimensions are array of input dimensions, it is sorted in increasing order
    // every time at the beginning we erase first element from it (not good idea to use vector for this purpose, but luckily it is small enough)
    // then we use this array for tads building, every time while recursion the number of built tads becomes bigger 
    dimensions.erase(dimensions.begin());       
    // build tad basing on output array, also create auxiliary arrays pointing on required output array ranges
    shape::TAD tadOut(output.getShapeInfo(), dimensions.data(), dimensions.size());
    tadOut.createTadOnlyShapeInfo();
    tadOut.createOffsets();
    NDArray<T> subArrOut(output.getBuffer(), tadOut.tadOnlyShapeInfo, output.getWorkspace());
    NDArray<T> subArr(output.getBuffer(), tadOut.tadOnlyShapeInfo, output.getWorkspace());
    // build tad basing on input array, also create auxiliary array pointing on required input array range
    shape::TAD tadIn(input.getShapeInfo(), dimensions.data(), dimensions.size());
    tadIn.createTadOnlyShapeInfo();
    tadIn.createOffsets();
    NDArray<T> subArrIn(input.getBuffer(), tadIn.tadOnlyShapeInfo, output.getWorkspace());
    // these indices take into account recursion and always point to actual tads numbers
    if (input.rankOf() > 1 && output.rankOf() > 1) {// only for non-vector cases
        outIdx = outIdx * output.sizeAt(dim + 1);
        inIdx = inIdx * input.sizeAt(dim + 1);
    }
    // current input tad number, we add to it unity in a loop
    int k = -1;
    // loop through current dimension
    for(int i = 0; i < output.sizeAt(dim); ++i) {
        // corresponds to outer range (relevant indices are absent in input)                        
        leftOffset = (int)paddings(dim, 0);
        if(i < leftOffset || i >= (input.sizeAt(dim) + leftOffset))
            continue;

        // increase input tads number
        ++k;
        // recursion condition allows for the fact that tad can't reduce to scalar
        if(dim < input.rankOf() - 2)
            recursiveLoopForPad(mode, input, paddings, output, dimensions, dim + 1, inIdx + k, outIdx + i, padValue);
        else if (paddings.sizeAt(0) > dim + 1){
            leftOffset = (int)paddings(dim + 1, 0);
            // shift buffers pointers to actual element position
            if (output.rankOf() > 1) {
                subArrOut.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + i]);
                subArrIn.setBuffer(input.getBuffer() + tadIn.tadOffsets[inIdx + i - (int) paddings(dim, 0)]);
            }
            else {
                subArrOut(i) = subArrIn(i - leftOffset);
            }
            // most inner loop, corresponds to last dim = rank-1
            switch (mode) {
                case 0:             // CONSTANT mode                    
                    for(int j = 0; j < subArrOut.lengthOf(); ++j)                   
                            if(j < leftOffset || j >= (subArrIn.lengthOf() + leftOffset) )                  // firstly fill with zeros outer ranges
                                subArrOut(j) = (T)0.;
                            else
                                subArrOut(j) = subArrIn(j - leftOffset);   // fill middle with elements of input array
                    break;

                case 1:             // REFLECT mode                 
                    for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side 
                        subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar(j));                       
                    for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
                        subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar(j));                   
                    for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
                        subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar(subArrOut.lengthOf() - j - 1));
                    break;

                case 2:             // SYMMETRIC mode               
                    for(int j = 1;  j <= leftOffset; ++j)                                               // fill firstly left side 
                        subArrOut.putIndexedScalar(leftOffset - j, subArrIn.getIndexedScalar(j-1));                             
                    for(int j = 0; j < subArrIn.lengthOf(); ++j)                                        // fill middle
                        subArrOut.putIndexedScalar(leftOffset + j, subArrIn.getIndexedScalar(j));                   
                    for(int j = (subArrOut.lengthOf() - leftOffset); j < subArrOut.lengthOf(); ++j)     // fill right side
                        subArrOut.putIndexedScalar(j, subArrIn.getIndexedScalar(subArrOut.lengthOf() - j));     
                    break;
            }
        }
        else {

             if (mode == 0 && input.rankOf() < 2)
                 subArrOut(i) = subArrIn(i - leftOffset);   // fill middle with elements of input array
        }   
    }   
    // populate sub-array formed previously 
    leftOffset = (int)paddings(dim,0);       
    switch (mode) {
        case 0:         // CONSTANT mode
            for(int j = 1;  j <= leftOffset; ++j) {
                // fill left side with padValue
                if (output.rankOf() > 1) {
                    subArrOut.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + leftOffset - j]);
                    subArrOut.assign(padValue);
                }
                else {
                    subArrOut(j - 1) = padValue;
                }
            }
//            output.printIndexedBuffer("Output at");
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill left side with zeros
                if (output.rankOf() > 1) {
                    subArrOut.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + j]);
                    subArrOut.assign(padValue);
                }
                else {
                    subArrOut(j) = padValue;
                }
            }
            break;

        case 1:         // REFLECT mode 
            for(int j = 1;  j <= leftOffset; ++j) {                                                     // fill left side 
                subArr.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + leftOffset + j]);
                subArrOut.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + leftOffset - j]);
                subArrOut.assign(&subArr);
            }               
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill right side
                subArr.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + output.sizeAt(dim) + leftOffset - 1 - j]);
                subArrOut.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + j]);
                subArrOut.assign(&subArr);              
            }   
            break;

        case 2:         // SYMMETRIC mode   
            for(int j = 1;  j <= leftOffset; ++j) {                                                     // fill left side
                subArr.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + leftOffset + j - 1]);
                subArrOut.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + leftOffset - j]);
                subArrOut.assign(&subArr);
            }           
            for(int j = (output.sizeAt(dim) - leftOffset); j < output.sizeAt(dim); ++j) {       // fill right side
                subArr.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + output.sizeAt(dim) + leftOffset - j]);
                subArrOut.setBuffer(output.getBuffer() + tadOut.tadOffsets[outIdx + j]);
                subArrOut.assign(&subArr);      
            }
            break;
    }
}
*/

////////////////////////////////////////////////////////////////////////
template<typename T>
void invertPermutation(const NDArray<T>& input, NDArray<T>& output) {

    std::set<int> uniqueElems;
    const int length = input.lengthOf();    

// #pragma omp parallel for if(length > Environment::getInstance()->elementwiseThreshold()) schedule(static)         
    for(int i = 0; i < length; ++i) {
        
        int elem = (int)input(i);
 
        if(!uniqueElems.insert(elem).second)        // this operation forbids us to use #pragma omp
            throw std::runtime_error("helpers::invertPermutation function: input array contains duplicates !");
            
        if(elem < 0 || elem > length - 1)
            throw  std::runtime_error("helpers::invertPermutation function: element of input array is out of range (0, length-1) !");

        output(elem) = i;
    }
}

////////////////////////////////////////////////////////////////////////
template<typename T>
void gatherND(NDArray<T>& input, NDArray<T>& indices, NDArray<T>& output) {

    if (input.ordering() != 'c') 
        input.streamline('c');

    if (indices.ordering() != 'c')
        indices.streamline('c');

    const int rankIn     = input.rankOf();
    const int rankInd    = indices.rankOf();
    const int lastIndDim = indices.sizeAt(-1);
    
    std::vector<int> tadDims(rankIn - lastIndDim);
    std::iota(tadDims.begin(), tadDims.end(), rankInd-1);
    ResultSet<T>* innerMostOut = output.allTensorsAlongDimension(tadDims); 

    ResultSet<T>* innerMostInd = indices.allTensorsAlongDimension({rankInd-1}); 
    
    std::iota(tadDims.begin(), tadDims.end(), lastIndDim);
    ResultSet<T>* innerMostIn = input.allTensorsAlongDimension(tadDims);

    Nd4jLong* outerShapeInfo = nullptr;
    ALLOCATE(outerShapeInfo, input.getWorkspace(), shape::shapeInfoLength(lastIndDim), Nd4jLong);
    outerShapeInfo[0] = lastIndDim;
    for(int i = 1; i <= lastIndDim; ++i)
        outerShapeInfo[i] = input.sizeAt(i-1);
    shape::updateStrides(outerShapeInfo, input.ordering());

    Nd4jLong idx[MAX_RANK];

    for(int i = 0; i < innerMostInd->size(); ++i) {
                
        NDArray<T>* idxSubArr = innerMostInd->at(i);        
        
        for(int j = 0; j < lastIndDim; ++j) {
            if((int)(*idxSubArr)(j) >= input.sizeAt(j))
                throw std::runtime_error("helpers::gatherND function: indices array contains wrong elements, each element must be smaller than corresponding dimension of input array !");
            idx[j] = (*idxSubArr)(j);
        }
                
        auto currentInd0 = shape::getOffset(0, shape::shapeOf(outerShapeInfo), shape::stride(outerShapeInfo), idx, lastIndDim);

        if(rankIn != lastIndDim) {
            NDArray<T>* outSubArr = innerMostOut->at(i);
            outSubArr->assign(innerMostIn->at(currentInd0));
        }
        else
            output(i) = input(currentInd0);
    }

    delete innerMostInd;
    delete innerMostIn;
    delete innerMostOut;
    RELEASE(outerShapeInfo, input.getWorkspace());    
}


////////////////////////////////////////////////////////////////////////
template<typename T>
void gather(NDArray<T>* input, const NDArray<T>* indices, NDArray<T>* output, const std::vector<int>& intArgs) {

    int axis = intArgs.size() > 0 ? intArgs[0] : 0;
    const int inputRank = input->rankOf();
    if(axis < 0)
        axis += inputRank;

    const int numOfIntArgs = intArgs.size();

    if (indices != nullptr) {        

        for(int i = 0; i < indices->lengthOf(); ++i)
            if((int)(*indices)(i) >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: indices array contains wrong elements, each element must be smaller than corresponding dimension of input array !");
    
        // first case: indices consist of only one scalar
        if(indices->isScalar()) {
            std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {axis});
            shape::TAD tad(input->getShapeInfo(), dimensions.data(), dimensions.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();
            NDArray<T> tadArr(input->getBuffer() + tad.tadOffsets[(int)(*indices)(0.)], tad.tadOnlyShapeInfo);
            output->assign(&tadArr);
        }
        else if (input->rankOf() == 1 && indices->isVector()) {
            // special case
#pragma omp parallel for if(indices->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
            for (int e = 0; e < indices->lengthOf(); e++)
                (*output)(e) = (*input)((*indices)(e));
        }
        // second case: indices is vector
        else if(indices->isVector()) {      
            ResultSet<T>* listOut = output->allTensorsAlongDimension(ShapeUtils<T>::evalDimsToExclude(output->rankOf(), {axis}));
            ResultSet<T>* listIn  = input->allTensorsAlongDimension(ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis}));
#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)             
            for(int i = 0; i < listOut->size(); ++i)
                listOut->at(i)->assign(listIn->at((int)(*indices)(i)));
            delete listOut;
            delete listIn;
        }
        // third case: indices is usual n-dim array
        else {
            std::vector<int> dimsOut(indices->rankOf());
            std::iota(dimsOut.begin(), dimsOut.end(), axis);   // fill with axis, axis+1, ... indices->rankOf()-1
            std::vector<int> temp1 = ShapeUtils<T>::evalDimsToExclude(output->rankOf(), dimsOut);
            std::vector<int> temp2 = ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis});
            ResultSet<T>* listOut = output->allTensorsAlongDimension(temp1);
            ResultSet<T>* listIn = input->allTensorsAlongDimension(temp2 );
#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
            for(int i = 0; i < listOut->size(); ++i)
                listOut->at(i)->assign(listIn->at((int)(*indices)(i)));
            delete listOut;
            delete listIn;
        }
    } 
    else {          // in this case always (numOfIntArgs > 1) !!!
        
        for(int i = 1; i < numOfIntArgs; ++i)
            if(intArgs[i] >= input->sizeAt(axis))
                throw std::runtime_error("helpers::gather function: some of input indexes is larger than corresponding shape of input array !");

        // we only allow scalar/vector case here
        if (numOfIntArgs == 2) {
            // scalar case
            std::vector<int> dimensions = ShapeUtils<T>::evalDimsToExclude(input->rankOf(), {axis});
            shape::TAD tad(input->getShapeInfo(), dimensions.data(), dimensions.size());
            tad.createTadOnlyShapeInfo();
            tad.createOffsets();
            NDArray<T> tadArr(input->getBuffer() + tad.tadOffsets[intArgs[1]], tad.tadOnlyShapeInfo);
            output->assign(&tadArr);
        } else {
            // vector case
            ResultSet<T>* listOut = output->allTensorsAlongDimension(ShapeUtils<T>::evalDimsToExclude(output->rankOf(), {axis}));
            ResultSet<T>* listIn  = input->allTensorsAlongDimension(ShapeUtils<T>::evalDimsToExclude(input->rankOf(),  {axis}));

            // that's fine, since we know that number of iArgs matches number of elements in listOut
#pragma omp parallel for if(listOut->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)     
            for(int i = 0; i < listOut->size(); ++i)
                listOut->at(i)->assign(listIn->at(intArgs[i+1]));
            delete listOut;
            delete listIn;
        }
    }    
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void eye(NDArray<T>& output) {

    const int rank = output.rankOf();
    ResultSet<T>* arrs = output.allTensorsAlongDimension({rank-2, rank-1});

#pragma omp parallel for if(arrs->size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
    for(int i = 0; i < arrs->size(); ++i)
        arrs->at(i)->setIdentity();
    
    delete arrs;    
    
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void scatterUpdate(NDArray<T>& operand, NDArray<T>& updates, const std::vector<int>* intArgs) {

    int opCode = (*intArgs)[0];
    int dimSize = (*intArgs)[1];    
    unsigned long e;
    unsigned long limg = 2 + dimSize;
    std::vector<int> tadDimension(limg-2);
    for (e = 2; e < limg; e++)
        tadDimension[e-2] = (*intArgs)[e];

    // increasing counter to skip numIndices
    e++;
    std::vector<int> indices;
    std::vector<int> indicesU;
    int cnt = 0;
    for (; e < intArgs->size(); e++) {
        indices.push_back((*intArgs)[e]);
        indicesU.push_back(cnt++);
    }

    std::unique_ptr<ResultSet<T>> tadsOperand(operand.multipleTensorsAlongDimension(indices, tadDimension));
    std::unique_ptr<ResultSet<T>> tadsUpdate(updates.multipleTensorsAlongDimension(indicesU, tadDimension));

#pragma omp parallel for if(indices.size() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close) shared(tadsOperand, tadsUpdate)
    for (unsigned long x = 0; x < indices.size(); x++) {
                
        NDArray<T> *tad = tadsOperand->at(x);
        NDArray<T> *tadUpdates = tadsUpdate->at(x);

        if (tad->lengthOf() != tadUpdates->lengthOf())
            continue;

        switch (opCode) {
            case 0:
                tad->template applyPairwiseTransform<simdOps::Add<T>>(tadUpdates, tad, nullptr);
                break;
            case 1:
                tad->template applyPairwiseTransform<simdOps::Subtract<T>>(tadUpdates, tad, nullptr);
                break;
            case 2:
                tad->template applyPairwiseTransform<simdOps::Multiply<T>>(tadUpdates, tad, nullptr);
                break;
            case 3:
                tad->template applyPairwiseTransform<simdOps::Divide<T>>(tadUpdates, tad, nullptr);
                break;
            case 4:
                tad->template applyPairwiseTransform<simdOps::ReverseSubtract<T>>(tadUpdates, tad, nullptr);
                break;
            case 5:
                tad->template applyPairwiseTransform<simdOps::ReverseDivide<T>>(tadUpdates, tad, nullptr);
                break;
            case 6:
                tad->template applyPairwiseTransform<simdOps::Copy<T>>(tadUpdates, tad, nullptr);
                break;
            default:
                continue;                 
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeMaxIndex(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {

    const Nd4jLong numArgs = inArrs.size();
    NDArray<T>* x = inArrs[0];    

#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
    for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        T max = -MAX_FLOAT;
        Nd4jLong idx = 0;
            
        for (int i = 0; i < numArgs; i++){
            
            T v = (*inArrs[i])(e);
            if (v > max) {
                max = v;
                idx = i;
            }
        }
        output(e) = idx;
    }
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeMax(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {
    
    const Nd4jLong numArgs = inArrs.size();
    NDArray<T> *x = inArrs[0];    

#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close)
     for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        T max = -MAX_FLOAT;
        for (int i = 0; i < numArgs; i++) { 
            T v = (*inArrs[i])(e);
            if (v > max)
                max = v;
        }
        output(e) = max;
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeAvg(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {
    
    const Nd4jLong numArgs = inArrs.size();
    const T factor = 1. / numArgs;
    NDArray<T> *x = inArrs[0];    
        
#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close)
    for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        T sum = 0.;
        for (int i = 0; i < numArgs; i++) { 
            T v = (*inArrs[i])(e);
            sum += v;
        }
        output(e) = sum * factor;
    }
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void mergeAdd(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output) {
    
    const Nd4jLong numArgs = inArrs.size();
    NDArray<T> *x = inArrs[0];    
        
#pragma omp parallel for if(x->lengthOf() > Environment::getInstance()->elementwiseThreshold()) schedule(guided) proc_bind(close)
    for (Nd4jLong e = 0; e < x->lengthOf(); e++) {
        
        T sum = 0.;
        
        for (int i = 0; i < numArgs; i++) 
            sum += (*inArrs[i])(e);;        

        output(e) = sum;
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void clipByNorm(NDArray<T>& input, NDArray<T>& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace) {    
        
    const int rank = input.rankOf();
    NDArray<T> norm2 = input.template reduceAlongDims<simdOps::Norm2<T>>(dimensions);

    if (isInplace) {

        if(norm2.lengthOf() == 1) {

            if(norm2(0.) > clipNorm)
                input *= (clipNorm / norm2(0.));
        }
        else {

            std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(rank, dimensions);
            const Nd4jLong numOfSubArrs = ShapeUtils<T>::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);            

#pragma omp parallel for schedule(guided) 
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {
                if (norm2(i) > clipNorm) {
                    
                    NDArray<T> inputSubArr  = input(i, dimsToExclude);
                    inputSubArr *= (clipNorm / norm2(i));
                }
            }
        }
    }
    else {
        
        if(norm2.lengthOf() == 1) {

            if(norm2(0.) > clipNorm)
                output.assign( input * (clipNorm / norm2(0.)));
            else
                output.assign( input );
        }
        else {
            
            std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(rank, dimensions);
            const Nd4jLong numOfSubArrs = ShapeUtils<T>::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
            std::vector<Nd4jLong> idxRanges(rank * 2);

#pragma omp parallel for schedule(guided) firstprivate(idxRanges)
            for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

                ShapeUtils<T>::evalIdxRangesForSubArr(i, input.getShapeInfo(), dimsToExclude, idxRanges.data());

                NDArray<T> outputSubArr = output(idxRanges);                
                NDArray<T> inputSubArr  = input(idxRanges);
                outputSubArr.assign(inputSubArr);
                
                if (norm2(i) > clipNorm) 
                    outputSubArr *= clipNorm / norm2(i);                
            }           
        }
    }
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void clipByNormBP(const NDArray<T>& input, const NDArray<T>& gradO, NDArray<T>& gradI /*output*/, const std::vector<int>& dimensions, const T clipNorm) {
    
    const int rank = input.rankOf();

    NDArray<T> norm2 = input.template reduceAlongDims<simdOps::Norm2<T>>(dimensions);

    if(norm2.lengthOf() == 1) {        

        const T N = norm2(0.);
        
        if(N > clipNorm) {            

            const T sumOfProd = (input * gradO).template reduceNumber<simdOps::Sum<T>>();    // reduce to scalar
            const T factor1 = static_cast<T>(1.f) / N;
            const T factor3 = factor1 / (N * N) ;                                            // 1 / (N*N*N)
            
            auto lambda = LAMBDA_TT(elem1, elem2, clipNorm, sumOfProd, factor1, factor3) { return clipNorm * (factor1 * elem2 - factor3 * elem1 * sumOfProd); };
            const_cast<NDArray<T>&>(input).applyPairwiseLambda(&gradO, lambda, &gradI);
        }
        else 
            gradI.assign(gradO);
    }
    else {
            
        std::vector<int> dimsToExclude = ShapeUtils<T>::evalDimsToExclude(rank, dimensions);
        const Nd4jLong numOfSubArrs = ShapeUtils<T>::getNumOfSubArrs(input.getShapeInfo(), dimsToExclude);
        std::vector<Nd4jLong> idxRanges(rank * 2);

#pragma omp parallel for schedule(guided) firstprivate(idxRanges)
        for(Nd4jLong i = 0; i < numOfSubArrs; ++i) {

            ShapeUtils<T>::evalIdxRangesForSubArr(i, input.getShapeInfo(), dimsToExclude, idxRanges.data());
            T N = norm2(i);

            NDArray<T> gradOSubArr = gradO(idxRanges);
            NDArray<T> gradISubArr = gradI(idxRanges);                
            
            if (N > clipNorm) {
                
                NDArray<T> inputSubArr = input(idxRanges);
                
                const T sumOfProd = (inputSubArr * gradOSubArr).template reduceNumber<simdOps::Sum<T>>();    // reduce to scalar
                const T factor1 = static_cast<T>(1.f) / N;
                const T factor3 = factor1 / (N * N) ;                                            // 1 / (N*N*N)

                auto lambda = LAMBDA_TT(elem1, elem2, clipNorm, sumOfProd, factor1, factor3) { return clipNorm * (factor1 * elem2 - factor3 * elem1 * sumOfProd); };
                inputSubArr.applyPairwiseLambda(&gradOSubArr, lambda, &gradISubArr);
            }
            else
                gradISubArr.assign(gradOSubArr);
        }           
    }
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void clipByAveraged(NDArray<T>& input, NDArray<T>& output, const std::vector<int>& dimensions, const T clipNorm, const bool isInplace) {
    
    if (dimensions.size() == 0) {
        // all-reduce
        T n2 = input.template reduceNumber<simdOps::Norm2<T>>() / input.lengthOf();        
        if (n2 <= clipNorm) {
            if (!isInplace)
                output.assign(input);
        } 
        else {
            const T factor = clipNorm / n2;
            auto lambda = LAMBDA_T(_x, factor) { return _x * factor; };
            input.applyLambda(lambda, &output);
        }
    } 
    else {
        // along dimension
        auto norm2 = input.template reduceAlongDims<simdOps::Norm2<T>>(dimensions, false);
        if (!isInplace)
                output.assign(input);
        auto tads = output.allTensorsAlongDimension(dimensions);
        // TODO: make this CUDA-compliant somehow
        for (int e = 0; e < tads->size(); e++) {
            T n2 = norm2.getScalar(e) / tads->at(e)->lengthOf();
            const T factor = clipNorm / n2;
            if (n2 > clipNorm) {
                auto lambda = LAMBDA_T(_x, factor) {return _x * factor;};
                tads->at(e)->applyLambda(lambda, &output);
            }
        }
        delete tads;
    }
}


//////////////////////////////////////////////////////////////////////////
template<typename T>
void mirrorPad(const NDArray<T>& input, const NDArray<T>& paddings, NDArray<T>& output, const int mode) {
    
    // mode:  0 - REFLECT, else - SYMMETRIC
    const int reflBorder = (bool)mode ? 1 : 0;
    const int symmBorder = (bool)mode ? 0 : 1;

    const int rank        = input.rankOf();
    const Nd4jLong outLen = output.lengthOf();
    const Nd4jLong inLen  = input.lengthOf();    

    if(rank <= 1) {

        const int leftSide  = static_cast<int>(paddings(static_cast<Nd4jLong>(0)));
        const int rightSide = static_cast<int>(paddings(static_cast<Nd4jLong>(1)));

//#pragma omp parallel for if(outLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(int i = 0; i < outLen; ++i) {
            
            for(int j = 0; j < leftSide; ++j) {
                int iindex = inLen - leftSide + symmBorder - j;
                if (iindex >= inLen) iindex = inLen - 1;
                if (iindex == inLen) iindex--;
                output(j) = input(iindex);

            }
            for(int j = 0; j < inLen; ++j)
                output(j + leftSide) = input(j);
            for(int j = 0; j < rightSide; ++j) {
                int iindex = inLen - 1 - symmBorder - j;
                if (iindex < 0) iindex = 0;
                if (iindex >= inLen) iindex = inLen - 1;
                output(leftSide + inLen + j) = input(iindex);
            }

        }  
    }
    else {

        std::vector<Nd4jLong> inIdx(rank), outIdx(rank);
#pragma omp parallel for if(outLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) firstprivate(inIdx, outIdx)
        for(int i = 0; i < outLen; ++i) {

            shape::ind2subC(rank, output.shapeOf(), i, outIdx.data());

            for(int j = 0; j < rank; ++j) {
            
                const int leftSide  = static_cast<int>(paddings(j, 0));

                if(outIdx[j] < leftSide) 
                    inIdx[j] = leftSide - outIdx[j] - reflBorder;

                else if(outIdx[j] >= leftSide && outIdx[j] < leftSide + input.sizeAt(j)) 
                    inIdx[j] = outIdx[j] - leftSide;

                else
                    inIdx[j] = 2 * input.sizeAt(j) + leftSide - outIdx[j] - 1 - symmBorder;                
            }
    
            Nd4jLong outOffset = shape::getOffset(0, output.shapeOf(), output.stridesOf(), outIdx.data(), rank);
            Nd4jLong inOffset  = shape::getOffset(0, input.shapeOf(),  input.stridesOf(),  inIdx.data(),  rank);
            output.buffer()[outOffset] = input.getBuffer()[inOffset];
        }
    }
}

//////////////////////////////////////////////////////////////////////////
template<typename T>
void concat(const std::vector<NDArray<T>*>& inArrs, NDArray<T>& output, const int axis) {

    const int numOfArrs = inArrs.size();
    bool allC = true;
    bool allScalar = true;
    bool allVectors = true;
    
    Nd4jLong lenOfFirstArr = inArrs[0]->lengthOf();

    //detect whether all arrays are c ordered or not
    //Also detect whether they are all scalars
    for(int i = 0; i < numOfArrs; i++) {
        allC &= (inArrs[i]->ordering() == 'c');
        allScalar &= (inArrs[i]->isScalar());
        allVectors &= (inArrs[i]->isRowVector() && inArrs[0]->lengthOf() == lenOfFirstArr);
    }

    //we are merging all scalars
    if(allScalar) {
        for(int i = 0; i < numOfArrs; i++) 
                 output.getBuffer()[i] = inArrs[i]->getBuffer()[0];
        return;
    }

    if(allC && axis == 0 && allVectors && output.ordering() == 'c') {
        
        if (numOfArrs >= 8) {

#pragma omp parallel for schedule(guided)
            for (int r = 0; r < numOfArrs; r++) {

                T *z = output.getBuffer() + (r * lenOfFirstArr);
                T *x = inArrs[r]->getBuffer();

#pragma omp simd
                for (Nd4jLong e = 0; e < lenOfFirstArr; e++)
                    z[e] = x[e];
            }
        } 
        else {
            int currBuffer = 0;
            int currBufferOffset = 0;
            for (int i = 0; i < output.lengthOf(); i++) {
                output.getBuffer()[i] = inArrs[currBuffer]->getBuffer()[currBufferOffset++];
                if (currBufferOffset >= inArrs[currBuffer]->lengthOf()) {
                    currBuffer++;
                    currBufferOffset = 0;
                }
            }
        }
        return;
    }
    
    const int rank  = inArrs[0]->rankOf();
    const int rank2 = 2*rank;
    std::vector<std::vector<Nd4jLong>> indices(numOfArrs, std::vector<Nd4jLong>(rank2,0));

    // take into account indices for first array
    indices[0][2 * axis + 1] = inArrs[0]->sizeAt(axis);

    // loop through the rest of input arrays
    for(int i = 1; i < numOfArrs; ++i) {
        indices[i][2 * axis]     = indices[i-1][2 * axis + 1];                                // index start from
        indices[i][2 * axis + 1] = indices[i-1][2 * axis + 1] + inArrs[i]->sizeAt(axis);      // index end with (excluding)
    }

// #pragma omp parallel for if(numOfArrs > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
#pragma omp parallel for schedule(guided)
    for(int i = 0; i < numOfArrs; ++i) {
        NDArray<T> temp = output(indices[i], true);
        temp.assign(inArrs[i]);
    }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
void tileBP(const NDArray<T>& gradO /*input*/, NDArray<T>& gradI /*output*/, const std::vector<Nd4jLong> reps) {

    T* gradIBuff      = gradI.getBuffer();
    const T* gradOBuff      = gradO.getBuffer();
    const Nd4jLong gradILen = gradI.lengthOf();
    const Nd4jLong gradOLen = gradO.lengthOf();  // gradOLen >= gradILen
    const Nd4jLong gradIEWS = nd4j::math::nd4j_abs<Nd4jLong>(gradI.ews());
    const Nd4jLong gradOEWS = gradO.ews();

    // initial zeroing of gradI content
    if(gradIEWS == 1)
        memset(gradIBuff, 0, gradILen * sizeof(T));
    else
#pragma omp parallel for schedule(static) proc_bind(close)
        for (int i = 0; i < gradILen * gradIEWS; i += gradIEWS)
            gradIBuff[i] = static_cast<T>(0.f);


    if(gradO.ordering() == 'c' && gradOEWS == 1) {
#pragma omp parallel for simd if(gradOLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i=0;  i<gradOLen; ++i)
            gradI(shape::subArrayIndex(gradO.getShapeInfo(), gradI.getShapeInfo(), i)) += gradOBuff[i];
    }
    else if(gradO.ordering() == 'c' && gradOEWS > 1) {
#pragma omp parallel for simd if(gradOLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided)
        for(Nd4jLong i=0;  i<gradOLen; ++i)
            gradI(shape::subArrayIndex(gradO.getShapeInfo(), gradI.getShapeInfo(), i)) += gradOBuff[i*gradOEWS];
    }
    else {
        Nd4jLong idx[MAX_RANK];
        Nd4jLong* gradOShape   = gradO.shapeOf();
        Nd4jLong* gradOStrides = gradO.stridesOf();
        const int gradORank    = gradO.rankOf();
#pragma omp parallel for simd if(gradOLen > Environment::getInstance()->elementwiseThreshold()) schedule(guided) private(idx)
        for(Nd4jLong i=0;  i<gradOLen; ++i) {
            shape::ind2subC(gradORank, gradOShape, i, gradOLen, idx);
            gradI(shape::subArrayIndex(gradO.getShapeInfo(), gradI.getShapeInfo(), i)) += gradOBuff[shape::getOffset(0, gradOShape, gradOStrides, idx, gradORank)];
        }
    }
}



template void triu<float>(const NDArray<float>& input, NDArray<float>& output, const int diagonal);
template void triu<float16>(const NDArray<float16>& input, NDArray<float16>& output, const int diagonal);
template void triu<double>(const NDArray<double>& input, NDArray<double>& output, const int diagonal);

template void triuBP<float>(const NDArray<float>& input, const NDArray<float>& gradO, NDArray<float>& gradI, const int diagonal);
template void triuBP<float16>(const NDArray<float16>& input, const NDArray<float16>& gradO, NDArray<float16>& gradI, const int diagonal);
template void triuBP<double>(const NDArray<double>& input, const NDArray<double>& gradO, NDArray<double>& gradI, const int diagonal);

template void trace<float>(const NDArray<float>& input, NDArray<float>& output);
template void trace<float16>(const NDArray<float16>& input, NDArray<float16>& output);
template void trace<double>(const NDArray<double>& input, NDArray<double>& output);

template void randomShuffle<float>(NDArray<float>& input, NDArray<float>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
template void randomShuffle<float16>(NDArray<float16>& input, NDArray<float16>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);
template void randomShuffle<double>(NDArray<double>& input, NDArray<double>& output, nd4j::random::RandomBuffer& rng, const bool isInplace);

// template void recursiveLoopForPad<float>(const int mode, NDArray<float>& input, const NDArray<float>& paddings, NDArray<float>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, float padValue);
// template void recursiveLoopForPad<float16>(const int mode, NDArray<float16>& input, const NDArray<float16>& paddings, NDArray<float16>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, float16 padValue);
// template void recursiveLoopForPad<double>(const int mode, NDArray<double>& input, const NDArray<double>& paddings, NDArray<double>& output, std::vector<int> dimensions, int dim, int inIdx, int outIdx, double padValue);

template void pad<float16>(const int mode, const NDArray<float16>& input, const NDArray<float16>& paddings, NDArray<float16>& output, const float16 padValue);
template void pad<float>(const int mode, const NDArray<float>& input, const NDArray<float>& paddings, NDArray<float>& output, const float padValue);
template void pad<double>(const int mode, const NDArray<double>& input, const NDArray<double>& paddings, NDArray<double>& output, const double padValue);

template void invertPermutation<float>(const NDArray<float>& input, NDArray<float>& output);
template void invertPermutation<float16>(const NDArray<float16>& input, NDArray<float16>& output);
template void invertPermutation<double>(const NDArray<double>& input, NDArray<double>& output);

template void gatherND<float>(NDArray<float>& input, NDArray<float>& indices, NDArray<float>& output);
template void gatherND<float16>(NDArray<float16>& input, NDArray<float16>& indices, NDArray<float16>& output);
template void gatherND<double>(NDArray<double>& input, NDArray<double>& indices, NDArray<double>& output);

template void gather<float>(NDArray<float>* input, const NDArray<float>* indices, NDArray<float>* output, const std::vector<int>& intArgs);
template void gather<float16>(NDArray<float16>* input, const NDArray<float16>* indices, NDArray<float16>* output, const std::vector<int>& intArgs);
template void gather<double>(NDArray<double>* input, const NDArray<double>* indices, NDArray<double>* output, const std::vector<int>& intArgs);

template void eye<float>(NDArray<float>& output);
template void eye<float16>(NDArray<float16>& output);
template void eye<double>(NDArray<double>& output);

template void scatterUpdate<float>(NDArray<float>& operand, NDArray<float>& updates, const std::vector<int>* intArgs);
template void scatterUpdate<float16>(NDArray<float16>& operand, NDArray<float16>& updates, const std::vector<int>* intArgs);
template void scatterUpdate<double>(NDArray<double>& operand, NDArray<double>& updates, const std::vector<int>* intArgs);

template void mergeMaxIndex<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeMaxIndex<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeMaxIndex<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void mergeMax<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeMax<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeMax<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void mergeAvg<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeAvg<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeAvg<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void mergeAdd<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output);
template void mergeAdd<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output);
template void mergeAdd<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output);

template void clipByNorm<float>(NDArray<float>& input, NDArray<float>& output, const std::vector<int>& dimensions, const float clipNorm, const bool isInplace);
template void clipByNorm<float16>(NDArray<float16>& input, NDArray<float16>& output, const std::vector<int>& dimensions, const float16 clipNorm, const bool isInplace);
template void clipByNorm<double>(NDArray<double>& input, NDArray<double>& output, const std::vector<int>& dimensions, const double clipNorm, const bool isInplace);

template void clipByNormBP<float>(const NDArray<float>& input, const NDArray<float>& gradO, NDArray<float>& gradI, const std::vector<int>& dimensions, const float clipNorm);
template void clipByNormBP<float16>(const NDArray<float16>& input, const NDArray<float16>& gradO, NDArray<float16>& gradI, const std::vector<int>& dimensions, const float16 clipNorm);
template void clipByNormBP<double>(const NDArray<double>& input, const NDArray<double>& gradO, NDArray<double>& gradI, const std::vector<int>& dimensions, const double clipNorm);

template void clipByAveraged<float>(NDArray<float>& input, NDArray<float>& output, const std::vector<int>& dimensions, const float clipNorm, const bool isInplace);
template void clipByAveraged<float16>(NDArray<float16>& input, NDArray<float16>& output, const std::vector<int>& dimensions, const float16 clipNorm, const bool isInplace);
template void clipByAveraged<double>(NDArray<double>& input, NDArray<double>& output, const std::vector<int>& dimensions, const double clipNorm, const bool isInplace);

template void mirrorPad<float>(const NDArray<float>& input, const NDArray<float>& paddings, NDArray<float>& output, const int mode);
template void mirrorPad<float16>(const NDArray<float16>& input, const NDArray<float16>& paddings, NDArray<float16>& output, const int mode);
template void mirrorPad<double>(const NDArray<double>& input, const NDArray<double>& paddings, NDArray<double>& output, const int mode);

template void tileBP<float>(const NDArray<float>& gradO, NDArray<float>& gradI, const std::vector<Nd4jLong> reps);
template void tileBP<float16>(const NDArray<float16>& gradO, NDArray<float16>& gradI, const std::vector<Nd4jLong> reps);
template void tileBP<double>(const NDArray<double>& gradO, NDArray<double>& gradI, const std::vector<Nd4jLong> reps);

template void concat<float>(const std::vector<NDArray<float>*>& inArrs, NDArray<float>& output, const int axis);
template void concat<float16>(const std::vector<NDArray<float16>*>& inArrs, NDArray<float16>& output, const int axis);
template void concat<double>(const std::vector<NDArray<double>*>& inArrs, NDArray<double>& output, const int axis);

}
}
}
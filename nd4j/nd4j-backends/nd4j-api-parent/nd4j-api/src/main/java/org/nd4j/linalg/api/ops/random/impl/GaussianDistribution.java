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

package org.nd4j.linalg.api.ops.random.impl;

import lombok.NonNull;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.BaseRandomOp;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * This Op generates normal distribution over provided mean and stddev
 *
 * @author raver119@gmail.com
 */
public class GaussianDistribution extends BaseRandomOp {
    private double mean;
    private double stddev;

    public GaussianDistribution(SameDiff sd, double mean, double stddev, long[] shape){
        super(sd, shape);
        this.mean = mean;
        this.stddev = stddev;
        this.extraArgs = new Object[] {this.mean, this.stddev};
    }

    public GaussianDistribution() {
        super();
    }

    /**
     * This op fills Z with random values within stddev..mean..stddev boundaries
     * @param z
     * @param mean
     * @param stddev
     */
    public GaussianDistribution(@NonNull INDArray z, double mean, double stddev) {
        init(z, z, z, z.lengthLong());
        this.mean = mean;
        this.stddev = stddev;
        this.extraArgs = new Object[] {this.mean, this.stddev};
    }


    public GaussianDistribution(@NonNull INDArray z, @NonNull INDArray means, double stddev) {
        if (z.lengthLong() != means.lengthLong())
            throw new IllegalStateException("Result length should be equal to provided Means length");

        if (means.elementWiseStride() < 1)
            throw new IllegalStateException("Means array can't have negative EWS");

        init(z, means, z, z.lengthLong());
        this.mean = 0.0;
        this.stddev = stddev;
        this.extraArgs = new Object[] {this.mean, this.stddev};
    }

    /**
     * This op fills Z with random values within -1.0..0..1.0
     * @param z
     */
    public GaussianDistribution(@NonNull INDArray z) {
        this(z, 0.0, 1.0);
    }

    /**
     * This op fills Z with random values within stddev..0..stddev
     * @param z
     */
    public GaussianDistribution(@NonNull INDArray z, double stddev) {
        this(z, 0.0, stddev);
    }

    @Override
    public int opNum() {
        return 6;
    }


    @Override
    public String opName() {
        return "distribution_gaussian";
    }

    @Override
    public boolean isExecSpecial() {
        return true;
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx op opName found for " +  opName());
    }

    @Override
    public String tensorflowName() {
        throw new NoOpNameFoundException("No tensorflow op opName found for " +  opName());
    }

    @Override
    public void setZ(INDArray z){
        //We want all 3 args set to z for this op
        this.x = z;
        this.y = z;
        this.z = z;
    }


    @Override
    public List<SDVariable> doDiff(List<SDVariable> f1) {
        return Collections.emptyList();
    }
}

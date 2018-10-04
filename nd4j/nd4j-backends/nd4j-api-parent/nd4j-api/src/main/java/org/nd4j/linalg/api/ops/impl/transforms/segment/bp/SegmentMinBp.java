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

package org.nd4j.linalg.api.ops.impl.transforms.segment.bp;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.linalg.api.ops.DynamicCustomOp;

/**
 * Segment min backprop operation
 *
 * @author Alex Black
 */
public class SegmentMinBp extends DynamicCustomOp {

    public SegmentMinBp(SameDiff sameDiff, SDVariable data, SDVariable segmentIds, SDVariable gradient) {
        super(null, sameDiff,  new SDVariable[] {data, segmentIds, gradient}, false);
    }

    public SegmentMinBp(){ }

    @Override
    public String opName(){
        return "segment_min_bp";
    }

}

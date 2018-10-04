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

package org.deeplearning4j.spark.parameterserver.networking.v1;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.optimize.solvers.accumulation.EncodingHandler;
import org.deeplearning4j.spark.parameterserver.networking.v1.messages.SilentUpdatesMessage;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.parameterserver.distributed.VoidParameterServer;

import java.util.concurrent.atomic.AtomicLong;

/**
 * This MessageHandler implementation does the same as EncodingHandler, plus additionally:
 * sends encoded messages over the wire + receives encoded messages from outer parties
 *
 * @author raver119@gmail.com
 */
@Slf4j
@Deprecated
public class WiredEncodingHandler extends EncodingHandler {
    protected AtomicLong updatesCounter = new AtomicLong(0);

    /**
     * This method builds new WiredEncodingHandler instance with initial encoding of 1e-3
     *
     */
    public WiredEncodingHandler() {
        super(1e-3);
    }

    /**
     * This method builds new WiredEncodingHandler instance
     *
     * @param threshold Initial encoding threshold
     */
    public WiredEncodingHandler(double threshold) {
        super(threshold);
    }

    /**
     * This method builds new WiredEncodingHandler instance
     *
     * @param threshold Initial encoding threshold
     * @param boundary
     */
    public WiredEncodingHandler(double threshold, Double boundary) {
        super(threshold, boundary);
    }

    /**
     * This method builds new WiredEncodingHandler instance
     *
     * @param threshold Initial encoding threshold
     * @param minThreshold Minimal encoding threshold (for threshold decay)
     * @param thresholdStep Decay step for threshold decay
     * @param stepTrigger Sparse/Dense ratio that will trigger decay step. In range 0..100
     * @param stepDelay Minimal number of iterations between decay steps
     * @param shakeFrequency How ofter we'll be sending dense updates with lower threshold
     */
    public WiredEncodingHandler(double threshold, double minThreshold, double thresholdStep, double stepTrigger,
                    int stepDelay, int shakeFrequency) {
        this(threshold, minThreshold, thresholdStep, stepTrigger, stepDelay, shakeFrequency, null);
    }

    /**
     * This method builds new WiredEncodingHandler instance
     *
     * @param threshold Initial encoding threshold
     * @param minThreshold Minimal encoding threshold (for threshold decay)
     * @param thresholdStep Decay step for threshold decay
     * @param stepTrigger Sparse/Dense ratio that will trigger decay step. In range 0..100
     * @param stepDelay Minimal number of iterations between decay steps
     * @param shakeFrequency How ofter we'll be sending dense updates with lower threshold
     * @param boundary
     */
    public WiredEncodingHandler(double threshold, double minThreshold, double thresholdStep, double stepTrigger,
                    int stepDelay, int shakeFrequency, Double boundary) {
        super(threshold, minThreshold, thresholdStep, stepTrigger, stepDelay, shakeFrequency, boundary);
    }

    /**
     * This method sends given message to all registered recipients
     *
     * @param message
     */
    @Override
    protected void sendMessage(INDArray message, int iterationNumber, int epochNumber) {
        // here we'll send our stuff to other executores over the wire
        // and let's pray for udp broadcast availability

        // Send this message away
        // FIXME: do something with unsafe duplication, which is bad and used ONLY for local spark
        try (MemoryWorkspace wsO = Nd4j.getMemoryManager().scopeOutOfWorkspaces()) {
            long updateId = updatesCounter.getAndIncrement();

            VoidParameterServer.getInstance().execDistributedImmediately(
                            new SilentUpdatesMessage(message.unsafeDuplication(), updateId));
        }


        // heere we update local queue
        super.sendMessage(message, iterationNumber, epochNumber);
    }
}

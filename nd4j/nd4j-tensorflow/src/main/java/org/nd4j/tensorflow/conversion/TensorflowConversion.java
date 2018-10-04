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

package org.nd4j.tensorflow.conversion;

import com.github.os72.protobuf351.InvalidProtocolBufferException;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.indexer.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.MetaGraphDef;
import org.tensorflow.framework.SignatureDef;
import org.tensorflow.framework.TensorInfo;

import java.io.IOException;
import java.lang.Thread;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

import static org.bytedeco.javacpp.tensorflow.*;

/**
 * Interop between nd4j {@link INDArray}
 * and {@link TF_Tensor}
 *
 * @author Adam Gibson
 */
public class TensorflowConversion {

    //used for passing to tensorflow: this dummy de allocator
    //allows us to use nd4j buffers for memory management
    //rather than having them managed by tensorflow
    private   static Deallocator_Pointer_long_Pointer calling;
    private static TensorflowConversion INSTANCE;

    /**
     * Get a singleton instance
     * @return
     */
    public static TensorflowConversion getInstance() {
        if(INSTANCE == null)
            INSTANCE = new TensorflowConversion();
        return INSTANCE;
    }


    private TensorflowConversion() {
        if(calling == null)
            calling = DummyDeAllocator.getInstance();

    }


    /**
     * Convert an {@link INDArray}
     * to a {@link TF_Tensor}
     * with zero copy.
     * Uses a direct pointer to the underlying ndarray's
     * data
     * @param ndArray the ndarray to use
     * @return the equivalent {@link TF_Tensor}
     */
    public TF_Tensor tensorFromNDArray(INDArray ndArray) {
       //we infer data type from the ndarray.databuffer()
        //for now we throw an exception
        if(ndArray.data() == null) {
           throw new IllegalArgumentException("Unable to infer data type from null databuffer");
       }

        if(ndArray.isView() || ndArray.ordering() != 'c') {
            ndArray = ndArray.dup('c');
        }


        long[] ndShape = ndArray.shape();
        long[] tfShape = new long[ndShape.length];
        for (int i = 0; i < ndShape.length; i++) {
            tfShape[i] = ndShape[i];
        }

        int type;
        DataBuffer data = ndArray.data();
        DataBuffer.Type dataType = data.dataType();
        switch (dataType) {
            case DOUBLE: type = DT_DOUBLE; break;
            case FLOAT:  type = DT_FLOAT;  break;
            case INT:    type = DT_INT32;  break;
            case HALF:   type = DT_HALF;   break;
            case COMPRESSED:
                CompressedDataBuffer compressedData = (CompressedDataBuffer)data;
                CompressionDescriptor desc = compressedData.getCompressionDescriptor();
                String algo = desc.getCompressionAlgorithm();
                switch (algo) {
                    case "FLOAT16": type = DT_HALF;   break;
                    case "INT8":    type = DT_INT8;   break;
                    case "UINT8":   type = DT_UINT8;  break;
                    case "INT16":   type = DT_INT16;  break;
                    case "UINT16":  type = DT_UINT16; break;
                    default: throw new IllegalArgumentException("Unsupported compression algorithm: " + algo);
                }
                break;
            case LONG: type = DT_INT64; break;
            default: throw new IllegalArgumentException("Unsupported data type: " + dataType);
        }

        try {
            Nd4j.getAffinityManager().ensureLocation(ndArray, AffinityManager.Location.HOST);
        } catch (Exception e) {
            // ND4J won't let us access compressed data in GPU memory, so we'll leave TensorFlow do the conversion instead
            ndArray.getDouble(0); // forces decompression and data copy to host
            data = ndArray.data();
            dataType = data.dataType();
            switch (dataType) {
                case DOUBLE: type = DT_DOUBLE; break;
                case FLOAT:  type = DT_FLOAT;  break;
                case INT:    type = DT_INT32;  break;
                case LONG:   type = DT_INT64;  break;
                default: throw new IllegalArgumentException("Unsupported data type: " + dataType);
            }
        }


        LongPointer longPointer = new LongPointer(tfShape);

        TF_Tensor tf_tensor = TF_NewTensor(
                type,
                longPointer,
                tfShape.length,
                data.pointer(),
                data.length() * data.getElementSize(),
                calling,null);

        return tf_tensor;

    }

    /**
     * Convert a {@link INDArray}
     * to a {@link TF_Tensor}
     *  using zero copy.
     *  It will use the underlying
     *  pointer with in nd4j.
     * @param tensor the tensor to use
     * @return
     */
    public INDArray ndArrayFromTensor(TF_Tensor tensor) {
        int rank = TF_NumDims(tensor);

        int[] ndShape;
        if (rank == 0) {
            // scalar
            ndShape = new int[] { 1 };
        } else {
            ndShape = new int[rank];
            for (int i = 0; i < ndShape.length; i++) {
                ndShape[i] = (int) TF_Dim(tensor,i);
            }
        }

        int tfType = TF_TensorType(tensor);
        DataBuffer.Type nd4jType = typeFor(tfType);

        int length = ArrayUtil.prod(ndShape);
        Pointer pointer = TF_TensorData(tensor).capacity(length);
        Indexer indexer = indexerForType(nd4jType,pointer);
        DataBuffer d = Nd4j.createBuffer(indexer.pointer(),nd4jType,length,indexer);
        INDArray array = Nd4j.create(d,ndShape);
        Nd4j.getAffinityManager().tagLocation(array, AffinityManager.Location.HOST);
        return array;
    }




    private Indexer indexerForType(DataBuffer.Type type,Pointer pointer) {
        switch(type) {
            case DOUBLE: return DoubleIndexer.create(new DoublePointer(pointer));
            case FLOAT: return FloatIndexer.create(new FloatPointer(pointer));
            case INT: return IntIndexer.create(new IntPointer(pointer));
            case LONG: return LongIndexer.create(new LongPointer(pointer));
            default: throw new IllegalArgumentException("Illegal type " + type);
        }
    }

    private DataBuffer.Type typeFor(int tensorflowType) {
        switch(tensorflowType) {
            case DT_DOUBLE: return DataBuffer.Type.DOUBLE;
            case DT_FLOAT: return DataBuffer.Type.FLOAT;
            case DT_INT32: return DataBuffer.Type.LONG;
            case DT_INT64: return DataBuffer.Type.LONG;
            default: throw new IllegalArgumentException("Illegal type " + tensorflowType);
        }
    }

    /**
     * Get an initialized {@link TF_Graph}
     * based on the passed in file
     * (the file must be a binary protobuf/pb file)
     * The graph will be modified to be associated
     * with the device associated with this current thread.
     *
     * Depending on the active {@link Nd4j#getBackend()}
     * the device will either be the gpu pinned to the current thread
     * or the cpu
     * @param filePath the path to the file to read
     * @return the initialized graph
     * @throws IOException
     */
    public TF_Graph loadGraph(String filePath, TF_Status status) throws IOException {
        byte[] bytes = Files.readAllBytes(Paths.get(filePath));
        return loadGraph(bytes, status);
    }

    /**
     * Infers the device for the given thread
     * based on the {@link Nd4j#getAffinityManager()}
     * Usually, this will either be a gpu or cpu
     * reserved for the current device.
     * You can think of the "current thread"
     * as a worker. This is mainly useful with multiple gpus
     * @return
     */
    public static String defaultDeviceForThread() {
        Integer deviceForThread = Nd4j.getAffinityManager().getDeviceForThread(Thread.currentThread());
        String deviceName = null;
        //gpu
        if(Nd4j.getBackend().getClass().getName().contains("JCublasBackend")) {
            deviceName = "/device:gpu:" + deviceForThread;
        }
        else {
            deviceName = "/device:cpu:" + deviceForThread;
        }


        return deviceName;
    }



    /**
     * Get an initialized {@link TF_Graph}
     * based on the passed in byte array content
     * (the content must be a binary protobuf/pb file)
     * The graph will be modified to be associated
     * with the device associated with this current thread.
     *
     * Depending on the active {@link Nd4j#getBackend()}
     * the device will either be the gpu pinned to the current thread
     * or the content
     * @param content the path to the file to read
     * @return the initialized graph
     * @throws IOException
     */

    public TF_Graph loadGraph(byte[] content, TF_Status status) {
        byte[] toLoad = content;
        TF_Buffer graph_def = TF_NewBufferFromString(new BytePointer(toLoad), content.length);
        TF_Graph graphC = TF_NewGraph();
        TF_ImportGraphDefOptions opts = TF_NewImportGraphDefOptions();
        TF_GraphImportGraphDef(graphC, graph_def, opts, status);
        if (TF_GetCode(status) != TF_OK) {
            throw new IllegalStateException("ERROR: Unable to import graph " + TF_Message(status).getString());
        }


        TF_DeleteImportGraphDefOptions(opts);

        return graphC;
    }

    public TF_Session loadSavedModel(String savedModelPath, TF_SessionOptions options, TF_Buffer runOptions,
            String tag, String key, TF_Graph graph, Map<String, String> inputsMap, Map<String, String> outputsMap, TF_Status status) {
        TF_Buffer metaGraph = TF_Buffer.newBuffer();
        TF_Session session = TF_LoadSessionFromSavedModel(options, runOptions, new BytePointer(savedModelPath),
                new BytePointer(tag), 1, graph, metaGraph, status);
        if (TF_GetCode(status) != TF_OK) {
            throw new IllegalStateException("ERROR: Unable to import model " + TF_Message(status).getString());
        }

        MetaGraphDef metaGraphDef;
        try {
            metaGraphDef = MetaGraphDef.parseFrom(metaGraph.data().capacity(metaGraph.length()).asByteBuffer());
        } catch (InvalidProtocolBufferException ex) {
            throw new IllegalStateException("ERROR: Unable to import model " + ex);
        }
        Map<String, SignatureDef> signatureDefMap = metaGraphDef.getSignatureDefMap();
        SignatureDef signatureDef = signatureDefMap.get(key);

        Map<String, TensorInfo> inputs = signatureDef.getInputsMap();
        for (Map.Entry<String, TensorInfo> e : inputs.entrySet()) {
            inputsMap.put(e.getKey(), e.getValue().getName());
        }

        Map<String, TensorInfo> outputs = signatureDef.getOutputsMap();
        for (Map.Entry<String, TensorInfo> e : outputs.entrySet()) {
            outputsMap.put(e.getKey(), e.getValue().getName());
        }

        return session;
    }
}

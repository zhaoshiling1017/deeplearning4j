package org.nd4j.linalg.api.ops.impl.shape;

import com.sun.tools.javac.util.ArrayUtils;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.NoOpNameFoundException;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.*;

/**
 * Stacks n input tensors of same shape to tensor of rank n + 1.
 *
 * @author farizrahman4u@gmail.com
 */
public class ParallelStack extends DynamicCustomOp {

    public ParallelStack() {
    }

    public ParallelStack(SameDiff sameDiff, SDVariable[] values) {
        super(null, sameDiff, values, false);
    }

    @Override
    public String onnxName() {
        throw new NoOpNameFoundException("No onnx opName found for " + opName());
    }

    @Override
    public String tensorflowName() {
        return "parallel_stack";
    }


    @Override
    public String opName() {
        return "parallel_stack";
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);
    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        throw new UnsupportedOperationException("No analog found for onnx for " + opName());
    }


    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        ret.put(tensorflowName(), map);
        return ret;
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable[] grads = args();
        int numInputs = grads.length;
        for(int i=0; i<numInputs; i++){
            SDVariable grad_in = f().gather(i_v.get(0), 0, new int[]{i});
            grad_in = f().squeeze(grad_in, 0);
            grads[0] = grads[0].mul(grad_in);
        }
        return Arrays.asList(grads);
    }

}

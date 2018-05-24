package org.nd4j.linalg.api.ops.impl.shape;

import lombok.NoArgsConstructor;
import lombok.val;
import onnx.OnnxProto3;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.imports.descriptors.properties.PropertyMapping;
import org.nd4j.imports.graphmapper.onnx.OnnxGraphMapper;
import org.nd4j.imports.graphmapper.tf.TFGraphMapper;
import org.nd4j.linalg.api.ops.DynamicCustomOp;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.ArrayUtil;
import org.tensorflow.framework.AttrValue;
import org.tensorflow.framework.GraphDef;
import org.tensorflow.framework.NodeDef;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Gather op
 */
@NoArgsConstructor
public class Gather extends DynamicCustomOp {

    protected int[] broadcast;
    protected int axis = 0;


    public Gather(SameDiff sameDiff, SDVariable input, int axis, int[] broadcast, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input}, inPlace);

        addIArgument(axis);
        addIArgument(broadcast);
        this.axis = axis;
        this.broadcast = broadcast;
    }

    public Gather(SameDiff sameDiff, SDVariable input, SDVariable indices, int axis, boolean inPlace) {
        super(null, sameDiff, new SDVariable[] {input, indices}, inPlace);
        addIArgument(axis);
        this.axis = axis;
    }
    @Override
    public String onnxName() {
        return "Gather";
    }


    @Override
    public String[] tensorflowNames() {
        return new String[]{"Gather", "GatherV2"};
    }

    @Override
    public void initFromTensorFlow(NodeDef nodeDef, SameDiff initWith, Map<String, AttrValue> attributesForNode, GraphDef graph) {
        TFGraphMapper.getInstance().initFunctionFromProperties(nodeDef.getOp(), this, attributesForNode, nodeDef, graph);

    }

    @Override
    public void initFromOnnx(OnnxProto3.NodeProto node, SameDiff initWith, Map<String, OnnxProto3.AttributeProto> attributesForNode, OnnxProto3.GraphProto graph) {
        OnnxGraphMapper.getInstance().initFunctionFromProperties(node.getOpType(), this, attributesForNode, node, graph);
    }


    @Override
    public void resolvePropertiesFromSameDiffBeforeExecution() {
        super.resolvePropertiesFromSameDiffBeforeExecution();
        if (broadcast != null && numInputArguments() < 2) {
            if (numInputArguments() == 0) {
                addInputArgument(args()[0].getArr(), Nd4j.create(ArrayUtil.toFloats(broadcast)).reshape(broadcast.length));

            } else if (numInputArguments() == 1) {
                addInputArgument(Nd4j.create(ArrayUtil.toFloats(broadcast)));
            }

        }

        if (numIArguments() < 1) {
            addIArgument(axis);
        }

        if (numOutputArguments() < getDescriptor().getNumOutputs()) {
            val outputs = outputVariables();
            for (int i = 0; i < outputs.length; i++) {
                val output = outputs[i].getArr();
                addOutputArgument(output);
            }
        }


    }

    @Override
    public Map<String, Map<String, PropertyMapping>> mappingsForFunction() {
        Map<String, Map<String, PropertyMapping>> ret = new HashMap<>();
        Map<String, PropertyMapping> map = new HashMap<>();
        val broadcast = PropertyMapping.builder()
                .onnxAttrName("broadcast")
                .tfInputPosition(1)
                .propertyNames(new String[]{"broadcast"}).build();

        map.put("broadcast", broadcast);

        ret.put(tensorflowNames()[0], map);
        ret.put(onnxName(), map);

        Map<String, PropertyMapping> map2 = new HashMap<>();
        val broadcast2 = PropertyMapping.builder()
                .tfInputPosition(1)
                .propertyNames(new String[]{"broadcast"}).build();
        map2.put("broadcast", broadcast2);

        val axis2 = PropertyMapping.builder()
                .tfInputPosition(2)
                .propertyNames(new String[]{"axis"}).build();
        map2.put("axis", axis2);

        ret.put("GatherV2", map2);


        return ret;
    }

    @Override
    public String opName() {
        return "gather";
    }

    @Override
    public List<SDVariable> doDiff(List<SDVariable> i_v) {
        SDVariable grad_in = i_v.get(0);
        SDVariable grads = arg().mul(0);
        int dims[] = new int[grads.getShape().length];
        if (axis != 0){
            dims[0] = axis;
            for(int i=1; i<=axis;i++){
                dims[i] = i - 1;
            }
            for(int i=axis+1; i<dims.length; i++){
                dims[i] = i;
            }
            grads = f().permute(grads, dims);

        }
        SDVariable[] grads_unstacked = new SDVariable[(int)grads.getShape()[0]];
        for(int i=0; i<grads_unstacked.length;i++){
            grads_unstacked[0] = f().squeeze(f().gather(grads, 0, new int[]{i}), 0);
        }
        for(int i=0; i< broadcast.length; i++){
            SDVariable grad_in_slice = f().squeeze(f().gather(grad_in, 0, new int[]{i}), 0);
            grads_unstacked[broadcast[i]].addi(grad_in_slice);
        }
        grads = f().stack(grads_unstacked, 0);
        if (axis != 0) {
            // reverse permute
            int[] reverse_dims = new int[dims.length];
            for(int i=0; i< reverse_dims.length; i++){
                reverse_dims[dims[i]] = i;
            }
            grads = f().permute(grads, reverse_dims);
        }
        return Arrays.asList(grads);
    }
}

import os.path

import onnx
import numpy as np
import struct
import sys
import copy

def search_node_by_output_id(nodes, output_id: str):
    prev_node = None
    for node_id, node in enumerate(nodes):
        if output_id in node.output:
            prev_node = node
            break
    return prev_node

def get_prev_node(nodes, node):
    node_input_list = node.input
    prev_node_list = []
    for node_id, node in enumerate(nodes):
        for node_output in node.output:
            if node_output in node_input_list:
                prev_node_list.append(node)
    return prev_node_list

def get_next_node(nodes, node):
    node_output_list = node.output
    next_node_list = []
    for node_id, node in enumerate(nodes):
        for node_input in node.input:
            if node_input in node_output_list:
                next_node_list.append(node)
    return next_node_list

def get_conv_qdq_node(nodes, conv_node):
    # get conv input
    conv_input_id = conv_node.input[0]
    # print(conv_input_id)
    dequant_node = None
    quant_node = None
    # get dequant node by conv input
    for node_id, node in enumerate(nodes):
        if node.op_type == "DequantizeLinear" and conv_input_id in node.output:
            dequant_node = node
            break
    # get quant node by dequant input
    if dequant_node is not None:
        dequant_input_id = dequant_node.input[0]
        # print(dequant_input_id)
        for node_id, node in enumerate(nodes):
            if node.op_type == "QuantizeLinear" and dequant_input_id in node.output:
                quant_node = node
                break
    # print(dequant_node)
    # print(quant_node)
    return dequant_node, quant_node

def onnx_conv_horizon_fuse(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node
    # find qualified add op
    pattern = []
    for node_id, node in enumerate(graph.node):
        if node.op_type == "Add":
            avail_count = 0
            for input_id in node.input:
                prev_node = search_node_by_output_id(graph.node, input_id)
                # prev node must be BatchNorm or Conv
                if prev_node is not None:
                    if prev_node.op_type in ['BatchNormalization', 'Conv'] and \
                            len(prev_node.output) == 1:
                        avail_count += 1
            if avail_count == 2:
                pattern.append(node)
    # print(pattern)

    # process each add
    for add_node in pattern:
        prev_add_node_list = get_prev_node(nodes, add_node)
        # collect conv node
        conv_node_list = []
        for node in prev_add_node_list:
            if node.op_type == "BatchNormalization":
                prev_node_list = get_prev_node(nodes, node)
                assert len(prev_node_list) == 1 and prev_node_list[0].op_type == "Conv", \
                    "Conv horizon fusion pattern not match"
                conv_node_list.append(prev_node_list[0])
            else:
                conv_node_list.append(node)

        # print(conv_node_list)
        # collect qdq node
        qdq_node_list = []
        for node in conv_node_list:
            dequant_node, quant_node = get_conv_qdq_node(nodes, node)
            assert dequant_node is not None and quant_node is not None, "Conv horizon fusion pattern not match"
            qdq_node_list.extend((dequant_node, quant_node))

        # find scale node
        scale_node_list = []
        for qdq_node in qdq_node_list:
            scale_iput_id = qdq_node.input[1]
            for node in nodes:
                if scale_iput_id in node.output:
                    scale_node_list.append(node)
        # print(scale_node_list)
        # get max scale
        max = 0
        for scale_node in scale_node_list:
            val = np.frombuffer(scale_node.attribute[0].t.raw_data, dtype=np.float32)[0]
            print(val)
            if max < val:
                max = val
        # rewrite max scale
        for scale_node in scale_node_list:
            scale_node.attribute[0].t.raw_data = bytes(struct.pack("f", max))

        # check
        for scale_node in scale_node_list:
            val = np.frombuffer(scale_node.attribute[0].t.raw_data, dtype=np.float32)[0]
            print(val)

    return onnx_replica

def onnx_add_insert_qdqnode(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node
    # find qualified add op
    patterns = []
    for node_id, node in enumerate(graph.node):
        if node.op_type == "Add":
            same_input_node_list = []
            same_input = None
            for add_input in node.input:
                for other_id, other_node in enumerate(nodes):
                    if other_id != node_id:
                        for other_input in other_node.input:
                            if other_input == add_input:
                                same_input_node_list.append(other_node)
                                same_input = other_input
                                break
            # Find previous node of Add, which has two output, one is QuantizeLinear, other is Add
            if len(same_input_node_list) == 1 and same_input_node_list[0].op_type == 'QuantizeLinear':
                prev_add_node = search_node_by_output_id(nodes, same_input)
                dequant_node = get_next_node(nodes, same_input_node_list[0])[0]
                patterns.append((node, prev_add_node, same_input_node_list[0], dequant_node, same_input))
    print(patterns)
    for pattern in patterns:
        add_node, prev_add_node, quant_node, dequant_node, same_input = pattern
        dq_x, dq_s, dq_z = dequant_node.input
        new_quant_node = onnx.helper.make_node('QuantizeLinear',
                                                inputs=quant_node.input,
                                                outputs=[prev_add_node.name + "_Dequant"],
                                                name=prev_add_node.name + "_QuantizeLinear")
        new_dequant_node = onnx.helper.make_node('DequantizeLinear',
                                                inputs=[prev_add_node.name + "_Dequant", dq_s, dq_z],
                                                outputs=[prev_add_node.name + "_Add"],
                                                name=prev_add_node.name + "_DequantizeLinear")

        add_node.input.remove(same_input)
        add_node.input.append(prev_add_node.name + "_Add")
        for node_id, node in enumerate(graph.node):
            if node.name == prev_add_node.name:
                graph.node.insert(node_id + 1, new_quant_node)
                graph.node.insert(node_id + 2, new_dequant_node)

    return onnx_replica

        # new_dequant_node = onnx.helper.make_node('DequantizeLinear',
        #                                         inputs=quant_node.input,
        #                                         outputs=prev_add_node.output,
        #                                         name=prev_add_node.name + "_DequantizeLinear")


def onnx_remove_qdqnode(onnx_model):
    onnx_replica = copy.deepcopy(onnx_model)
    graph = onnx_replica.graph
    nodes = graph.node

    # demo for remove node with first input and output
    in_rename_map = {}
    scale_node_list = []
    zero_node_list = []
    activation_map = {}
    for node_id, node in enumerate(graph.node):
        if node.op_type == "QuantizeLinear":
            # node input
            in_name = node.input[0]
            scale_name = node.input[1]
            zero_name = node.input[2]
            # print(scale_name)
            # node output
            out_name = node.output[0]
            # record input, remove one node, set node's input to its next
            in_rename_map[out_name] = in_name
            scale_node_list.append(scale_name)
            zero_node_list.append(zero_name)
            # for i, node in enumerate(graph.node):
            #     if node.output[0] == scale_name:
            #         if len(node.attribute[0].t.dims) > 0:
            #             print(node.attribute[0].t.dims)
            #         graph.node.remove(nodes[i])
            # for i, node in enumerate(graph.node):
            #    if node.output[0] == zero_name:
            #        graph.node.remove(nodes[i])
            # record scale of activation
            for i, node in enumerate(graph.node):
                if node.output[0] == scale_name:
                    if len(node.attribute[0].t.dims) == 0:
                        # print(node.attribute[0].t.raw_data)
                        # print(np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32))
                        val = np.frombuffer(node.attribute[0].t.raw_data, dtype=np.float32)[0]
                        if in_name in activation_map.keys():
                            old_val = struct.unpack('!f', bytes.fromhex(activation_map[in_name]))[0]
                            # print("Already record, old {:.4f}, new {:.4f}".format(old_val, val))
                            if val > old_val:
                                activation_map[in_name] = struct.pack('>f', val).hex()
                        else:
                            activation_map[in_name] = struct.pack('>f', val).hex()
            # remove QuantizeLinear node
            graph.node.remove(nodes[node_id])


    # relink
    for node_id, node in enumerate(graph.node):
       for in_id, in_name in enumerate(node.input):
           if in_name in in_rename_map.keys():
               # set node input == removed node's input
               node.input[in_id] = in_rename_map[in_name]

    in_rename_map = {}
    # activation_map = {}
    for node_id, node in enumerate(graph.node):
       if node.op_type == "DequantizeLinear":
           in_name = node.input[0]
           scale_name = node.input[1]
           zero_name = node.input[2]
           # print(scale_name)
           out_name = node.output[0]
           in_rename_map[out_name] = in_name
           graph.node.remove(nodes[node_id])
           scale_node_list.append(scale_name)
           zero_node_list.append(zero_name)

    # relink
    for node_id, node in enumerate(graph.node):
       for in_id, in_name in enumerate(node.input):
           if in_name in in_rename_map.keys():
               node.input[in_id] = in_rename_map[in_name]

    nodes = graph.node
    for node_name in (scale_node_list + zero_node_list):
        for node_id, node in enumerate(graph.node):
            if node.name == node_name:
                # print("node input={}".format(node.input))
                # for node_input in node.input:
                #     print(node_input)
                #     graph.node.remove(node_input)
                graph.node.remove(nodes[node_id])

    for node_name in (scale_node_list + zero_node_list):
        for node_id, node in enumerate(graph.node):
            if node.output[0] == node_name:
                # print("node input={}".format(node.input))
                # for node_input in node.input:
                #     print(node_input)
                #     graph.node.remove(node_input)
                graph.node.remove(nodes[node_id])

    return onnx_replica, activation_map

def save_calib_cache_file(cache_file, activation_map, headline='TRT-8XXX-EntropyCalibration2\n'):
    with open(os.path.join(cache_file), 'w') as cfile:
        cfile.write(headline)
        for k, v in activation_map.items():
            cfile.write("{}: {}\n".format(k, v))

def get_remove_qdq_onnx_and_cache(onnx_file):
    model = onnx.load(onnx_file)
    # onnx_insert = onnx_add_insert_qdqnode(model)
    model_wo_qdq, activation_map = onnx_remove_qdqnode(model)
    onnx_name, onnx_dir = os.path.basename(onnx_file), os.path.dirname(onnx_file)
    onnx_new_name = onnx_name.replace('.onnx', '_remove_qdq.onnx')
    onnx.save(model_wo_qdq, os.path.join(onnx_dir, onnx_new_name))
    cache_name = onnx_new_name.replace('.onnx', '_add_insert_qdq_calibration.cache')
    save_calib_cache_file(os.path.join(onnx_dir, cache_name), activation_map)

if __name__ == '__main__':

    onnx_file = sys.argv[1]
    get_remove_qdq_onnx_and_cache(onnx_file)

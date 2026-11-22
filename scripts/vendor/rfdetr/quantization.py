# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0; see LICENSE.
# Modified for mqtt-camera-streamer: extracted graph-selection helpers only;
# quantization execution/calibration functions and package imports omitted.
"""RF-DETR 1.12.1 graph selection for attention/head-safe INT8 quantization."""
from __future__ import annotations
from typing import Any

#: The only ops quantized. See the module docstring for why the list is this short.
_QUANTIZED_OP_TYPES: tuple[str, ...] = ("MatMul", "Gemm")

#: Ops that end the walk back from a graph output. The walk excludes the ``MatMul``/``Gemm`` it stops at -- the last
#: linear layer of a head -- and nothing behind it; a ``Conv`` means it has left the heads for the backbone.
_HEAD_STOP_OP_TYPES: frozenset[str] = frozenset({*_QUANTIZED_OP_TYPES, "Conv"})

#: Elementwise ops allowed between an attention-score multiply and its ``Softmax`` -- the ``1/sqrt(d)`` scaling and a
#: constant bias. The walk back from a ``Softmax`` passes through one only when its other operand is a constant.
_SCORE_SCALING_OP_TYPES: frozenset[str] = frozenset({"Div", "Mul", "Add", "Sub"})

#: Upper bound on the hops any backward walk over the graph takes, so a malformed or cyclic graph cannot stall export.
_MAX_WALK_HOPS: int = 10


def _constant_names(graph: Any) -> set[str]:
    """Return the names of tensors whose value is fixed at export time.

    Args:
        graph: An ``onnx.GraphProto``.

    Returns:
        Initializer names plus the outputs of ``Constant`` nodes.
    """
    names = {initializer.name for initializer in graph.initializer}
    names.update(output for node in graph.node if node.op_type == "Constant" for output in node.output)
    return names


def _attention_score_matmuls(graph: Any, producer: dict[str, Any], constants: set[str]) -> set[str]:
    """Return the activation-by-activation multiplies whose result reaches a ``Softmax``.

    From each ``Softmax`` input the walk goes back through scaling ops (:data:`_SCORE_SCALING_OP_TYPES`) whose other
    operand is a constant, and stops at the first ``MatMul``/``Gemm``. That node is kept only when neither of its two
    data inputs is a constant -- the ``Q @ K^T`` shape. A weight multiply feeding a ``Softmax`` (a classifier, the
    deformable-attention weights) is an ordinary linear layer and stays quantizable.

    Args:
        graph: An ``onnx.GraphProto``.
        producer: Tensor name -> node producing it.
        constants: Names of constant tensors, from :func:`_constant_names`.

    Returns:
        Names of the attention-score multiplies.
    """
    excluded: set[str] = set()
    for node in graph.node:
        if node.op_type != "Softmax" or not node.input:
            continue
        tensor = node.input[0]
        for _ in range(_MAX_WALK_HOPS):
            source = producer.get(tensor)
            if source is None:
                break
            if source.op_type in _QUANTIZED_OP_TYPES:
                if source.name and not constants.intersection(source.input[:2]):
                    excluded.add(source.name)
                break
            dynamic = [name for name in source.input if name not in constants]
            if source.op_type not in _SCORE_SCALING_OP_TYPES or len(dynamic) != 1:
                break
            tensor = dynamic[0]
    return excluded


def _head_matmuls(graph: Any, producer: dict[str, Any]) -> set[str]:
    """Return the last ``MatMul``/``Gemm`` on each path back from a graph output.

    The walk goes back from every graph output through any op not in :data:`_HEAD_STOP_OP_TYPES` (activations, box
    refinement arithmetic, reshapes, concatenations) for at most :data:`_MAX_WALK_HOPS` hops. Each path ends at the
    first stop op; only a ``MatMul``/``Gemm`` terminal is excluded, so the layers behind it stay quantizable however
    many elementwise ops sit between the head and the output.

    Args:
        graph: An ``onnx.GraphProto``.
        producer: Tensor name -> node producing it.

    Returns:
        Names of the head multiplies.
    """
    excluded: set[str] = set()
    visited: set[str] = set()
    frontier = [output.name for output in graph.output]
    for _ in range(_MAX_WALK_HOPS):
        next_frontier: list[str] = []
        for tensor in frontier:
            node = producer.get(tensor)
            if node is None or tensor in visited:
                continue
            visited.add(tensor)
            if node.op_type not in _HEAD_STOP_OP_TYPES:
                next_frontier.extend(node.input)
            elif node.op_type in _QUANTIZED_OP_TYPES and node.name:
                excluded.add(node.name)
        frontier = next_frontier
    return excluded


def nodes_to_exclude(graph: Any) -> list[str]:
    """Return the names of nodes that must stay in float even among ``MatMul``/``Gemm``.

    Two groups, both found by walking the graph rather than matched by name, so a renamed or restructured export stays
    covered:

    * the attention-score multiplies -- an activation-by-activation ``MatMul`` reaching a ``Softmax`` directly or
      through constant scaling (``Q @ K^T / sqrt(d)``), where an 8-bit range quantizes the comparison the attention is
      about to make;
    * the last ``MatMul``/``Gemm`` on each path back from a graph output -- the detection heads' final layers, which
      emit box deltas and class logits directly, so quantizing them quantizes the answer. The walk passes through any
      number of elementwise ops (box refinement, sigmoid, reshapes) up to :data:`_MAX_WALK_HOPS` hops, and stops at a
      ``Conv``.

    Args:
        graph: An ``onnx.GraphProto``.

    Returns:
        Node names to pass to ``quantize_static(nodes_to_exclude=...)``, sorted and deduplicated.

    Examples:
        >>> import onnx
        >>> from onnx import helper, TensorProto
        >>> node = helper.make_node("MatMul", ["a", "b"], ["out"], name="head_matmul")
        >>> graph = helper.make_graph(
        ...     [node],
        ...     "g",
        ...     [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 1])],
        ...     [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 1])],
        ... )
        >>> nodes_to_exclude(graph)
        ['head_matmul']
    """
    producer = {output: node for node in graph.node for output in node.output}
    excluded = _attention_score_matmuls(graph, producer, _constant_names(graph))
    excluded |= _head_matmuls(graph, producer)
    return sorted(excluded)

from add import AddOperator
from gemm import GemmOperator
from matmul import MatMul
from relu import ReLUOperator
from sub import SubOperator
from concat import ConcatOperator
from reshape import ReshapeOperator
from flatten import FlattenOperator
from conv import ConvOperator

class OperatorFactory:
    def __init__(self):
        self.node_handlers = {
            'Gemm'          : GemmOperator,
            'Add'           : AddOperator,
            'MatMul'        : MatMul,
            'Relu'          : ReLUOperator,
            'Sub'           : SubOperator,
            'Concat'        : ConcatOperator,
            'Reshape'       : ReshapeOperator,
            'Flatten'       : FlattenOperator,
            'Conv'          : ConvOperator,
        }

    def create_operator(self, node, initializers):
        op_type = node['type']
        handler_class = self.node_handlers.get(op_type)
        if not handler_class:
            raise NotImplementedError(f"Operator '{op_type}' is not supported.")
        return handler_class(node, initializers)

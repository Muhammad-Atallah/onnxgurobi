from add_parser import AddParser
from gemm_parser import GemmParser
from matmul_parser import MatMulParser
from relu_parser import ReluParser
from sub_parser import SubParser
from concat_parser import ConcatParser
from reshape_parser import ReshapeParser
from flatten_parser import FlattenParser
from conv_parser import ConvParser
from unsqueeze_parser import UnsqueezeParser
from gather_parser import GatherParser
from maxpool_parser import MaxPoolParser
from averagepool_parser import AveragePoolParser
from dropout_parser import DropoutParser
from constant_parser import ConstantParser
from identity_parser import IdentityParser

class ParserFactory:
    def __init__(self):
        self.parsers = {
            'Add': AddParser,
            'Gemm': GemmParser,
            'MatMul': MatMulParser,
            'Relu': ReluParser,
            'Sub': SubParser,
            'Concat': ConcatParser,
            'Constant': ConstantParser,
            'Reshape': ReshapeParser,
            'Flatten': FlattenParser,
            'Conv': ConvParser,
            'Unsqueeze': UnsqueezeParser,
            'Gather': GatherParser,
            'MaxPool': MaxPoolParser,
            'AveragePool': AveragePoolParser,
            'Dropout': DropoutParser,
            'Identity': IdentityParser

        }

    def get_parser(self, op_type):
        parser = self.parsers.get(op_type)
        if not parser:
            raise NotImplementedError(f"Parser for operator '{op_type}' is not supported.")
        return parser

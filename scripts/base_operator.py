from abc import ABC, abstractmethod

class BaseOperator(ABC):
    def __init__(self, node, initializers):
        self.node = node
        self.initializers = initializers

    @abstractmethod
    def apply_constraints(self, gurobi_model, variables):
        pass

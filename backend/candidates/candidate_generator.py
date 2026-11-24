from .supervised.logistic_regression import get_candidate as model_1
from .supervised.random_forest import get_candidate as model_2
from .supervised.hist_gradient_boosting import get_candidate as model_3
from .supervised.decision_tree import get_candidate as model_4
from .supervised.extra_trees import get_candidate as model_5

def generate_candidates():
    return {
        "C1": model_1(),
        "C2": model_2(),
        "C3": model_3(),
        "C4": model_4(),
        "C5": model_5()
    }
from sklearn.ensemble import HistGradientBoostingClassifier

def get_candidate():
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=0.1, random_state=42)
    return model


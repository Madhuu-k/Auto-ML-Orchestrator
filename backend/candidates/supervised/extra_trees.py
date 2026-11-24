from sklearn.ensemble import ExtraTreesClassifier

def get_candidate():
    model = ExtraTreesClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=1)
    return model
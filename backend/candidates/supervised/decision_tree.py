from sklearn.tree import DecisionTreeClassifier

def get_candidate():
    model = DecisionTreeClassifier(max_depth=5, min_samples_split=2, random_state=42)
    return model
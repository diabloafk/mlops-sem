from sklearn.ensemble import RandomForestClassifier

def test_model():
    model = RandomForestClassifier()
    assert model is not None

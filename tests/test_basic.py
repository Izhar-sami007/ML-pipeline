from pipeline.preprocessing import load_data, prepare


def test_prepare():
    df = load_data('data/iris.csv')
    X_train, X_val, X_test, y_train, y_val, y_test = prepare(df)
    assert X_train.shape[0] > 0
    assert X_val.shape[0] > 0
    assert X_test.shape[0] > 0
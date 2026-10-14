from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from sklearn.pipeline import Pipeline

from house_ml.data.dataset import split_features_target
from house_ml.data.load import load_housing_data
from house_ml.data.split import stratified_split
from house_ml.features.preprocessing import build_preprocessor
from house_ml.models.model import build_model

def build_training_pipeline() -> Pipeline:
    '''
    Build complete preprocessing + model pipeline
    '''

    preprocessor = build_preprocessor()

    model = build_model()

    pipeline = Pipeline(
        steps = [
            ('preprocessing',
            preprocessor
            ),
            (
                'model',
                model
            )

        ]
    )

    return pipeline


def train_model(
        X_train,
        y_train
) -> Pipeline:
    
    pipeline = build_training_pipeline()

    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(
        model,
        X_test,
        y_test
) -> dict[str, float]:
    
    predictions = model.predict(X_test)

    metrics = {
        'rmse': root_mean_squared_error(y_test, predictions),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }

    return metrics


def main():

    print('Loading housing data...')

    housing = load_housing_data()

    print(f'Dataset shape: {housing.shape}')

    print('Creating train/test split...')

    train_set, test_set = stratified_split(housing)

    X_train, y_train = split_features_target(train_set)
    X_test, y_test = split_features_target(test_set)

    print(f'Training samples: {len(X_train)}')

    print(f'Test samples: {len(X_test)}')

    print('Building training pipeline...')

    pipeline = build_training_pipeline()

    print(pipeline)

    print('Training model...')

    pipeline.fit(X_train, y_train)

    print('Training complete.')

    print('Evaluating model...')

    metrics = evaluate_model(pipeline, X_test, y_test)

    print(
        f'RMSE: '
        f"{metrics['rmse']:.2f}"
    )

    print(
        'MAE: '
        f'{metrics['mae']:.2f}'
    )

    print(f'R^2: '
          f'{metrics['r2']:.4f}')


if __name__ == '__main__':
    main()
from ml_pipeline.playground_runtime import run_experiment


def build_dataset(tables, config):
    return tables[config['table']].copy()


if __name__ == '__main__':
    run_experiment(build_dataset)

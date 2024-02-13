import json
import pandas
import os
from collections import defaultdict

in_folder = 'analysis/results/'

df = pandas.DataFrame()
models = defaultdict(dict)
metric_names = []
if __name__ == '__main__':
    for model in sorted(os.listdir(in_folder)):
        models[model] = {}
        for file in os.listdir(in_folder + model):
            if file.endswith('.json') and file != 'structure.json':
                with open(in_folder + model + '/' + file) as f:
                    d = json.load(f)
                    d['model'] = model
                    for k, v in d.items():
                        models[model][k] = v
                        if k not in metric_names:
                            metric_names.append(k)

    for model, d in models.items():
        df = df.append(d, ignore_index=True)

    # best model is the one closer to dataset
    row_of_best = {'model': 'best'}
    for metric in metric_names:
        if metric == 'model':
            continue
        best_model = ''
        best_value = 1000000
        for model, d in models.copy().items():
            if model == 'test':
                continue
            if metric not in d:
                continue
            if metric not in models['test']:
                continue
            if abs(d[metric]-float(models['test'][metric])) < best_value:
                best_model = model
                best_value = abs(d[metric]-float(models['test'][metric]))

        row_of_best[metric] = best_model

    print(row_of_best)

    df = df.append(row_of_best, ignore_index=True)
    print(df)
    df = df.set_index('model')

    df.to_csv('analysis/results.csv')
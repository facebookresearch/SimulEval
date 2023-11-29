import os
import pandas as pd
import argparse
from pprint import pprint


def read_scores_from_folder(folder_path):
    score_file_path = os.path.join(folder_path, "scores.tsv")
    if os.path.isfile(score_file_path):
        with open(score_file_path, "r") as f:
            contents = [line.strip() for line in f.read().split("\n") if line.strip()]
        return contents
    else:
        return None


def read_scores_files(output_folder):
    all_contents = []

    if not os.path.isdir(output_folder):
        raise ValueError("Output folder does not exist")

    output_folder = os.path.abspath(output_folder)

    for folder in os.listdir(output_folder):
        folder_path = os.path.join(output_folder, folder)

        if os.path.isdir(folder_path):
            contents = read_scores_from_folder(folder_path)
            if contents:
                all_contents.append(contents)

    headers_list = []
    for contents in all_contents:
        if contents:
            header = contents[0].split()
            if not header:
                raise ValueError(f"Empty header in {contents}")
            headers_list.append(header)

    return all_contents, headers_list


def process_result(output_folder, metric_names):
    all_contents, headers_list = read_scores_files(output_folder)
    
    # Extracting headers from the first line of each "scores.tsv" file
    reference_header = headers_list[0]

    if metric_names is None:
        metric_names = reference_header
    common_metrics = set(metric_names).intersection(reference_header)

    if not common_metrics:
        raise ValueError("No common metrics found in the results")

    # Extracting scores for each metric
    scores = []
    for contents in all_contents:
        if contents:
            values = dict(zip(contents[0].split(), contents[1].split()))
            scores.append(values)

    df = pd.DataFrame(scores)

    # Fill NaN values with NaN
    df = df.fillna("NaN")
    filtered_df = df[df.columns[df.columns.isin(common_metrics)]]

    if len(common_metrics) == 1:
        metric_name = list(common_metrics)[0]
        filtered_df = filtered_df[filtered_df[metric_name] != 0.0]

    return filtered_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument(
        "--metrics", type=str, nargs="+", default=None, help="Metrics to be extracted"
    )
    args = parser.parse_args()

    df = process_result(args.output, args.metrics)
    pprint(df)

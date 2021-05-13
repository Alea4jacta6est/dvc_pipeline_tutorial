# MNIST training using DVC pipelines - tutorial

## How to reproduce the pipeline?

1. Clone this repository

2. Activate venv and `pip install -r requirements.txt` to install dependencies 

3. Use `dvc repro` to reproduce the pipeline (ask me for access to use data from remote)

4. Use `dvc metrics show` to get metrics of the last model

## Task description

Data governance for ML (DVC)

- Task description:
    1. Use dataset from previous task. Make initial setup using `Data version control` tool.
    2. Define 2-3 pipelines that would preprocess data in different ways (basic
    cleaning, scaling, aggregations, etc.). Each pipeline should be
    reproducible using `dvc repro` .
    3. Use some existing
    solution for your dataset, run experiment on the data using development
    environment from previous step and save metrics using `dvc metrics`
- Criteria:
    1. Pipelines defined in a simple, reproducible manner
    2. Following DVC best practices
    3. Code style / code quality tools used
    4. There is an existing remote from which one could pull data (use free tier of
    AWS/GCP, Google Drive, or any other that would be easy to share)
- Materials:
    - [DVC documentation and tutorials](https://dvc.org/doc/start)
    - [DVC video tutorials](https://www.youtube.com/channel/UC37rp97Go-xIX3aNFVHhXfQ/videos)
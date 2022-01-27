# https://towardsdatascience.com/a-complete-guide-to-building-a-docker-image-serving-a-machine-learning-system-in-production-d8b5b0533bde

FROM nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04
ARG POETRY_VERSION
ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt update \
    && apt install -y software-properties-common python3-pip \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt install -y python${PYTHON_VERSION}-dev \
    && apt clean && rm -rf /var/lib/apt/lists*

RUN python${PYTHON_VERSION} -m pip install "poetry==${POETRY_VERSION}" \
    && poetry config virtualenvs.create false

COPY poetry.lock* pyproject.toml ./
RUN poetry install --no-interaction --no-ansi

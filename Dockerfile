# https://www.reddit.com/r/MachineLearning/comments/13jud83/d_best_practices_to_dockerize_hugginface_hub/
# from https://discuss.huggingface.co/t/manually-downloading-models-in-docker-build-with-snapshot-download/19637/2
FROM continuumio/miniconda3
ARG APP_DIR="/home/app"
# install gcc for JIT compilation of transformers
RUN apt update \
    && apt install -y g++ \
    && rm -rf /var/lib/apt/lists/*
RUN groupadd -r app && useradd --no-log-init -r -g app app
#USER root
USER app

#RUN mkdir -p ${APP_DIR}
WORKDIR ${APP_DIR}
COPY ./ ${APP_DIR}
RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "--no-capture-output", "-n", "rag", "/bin/bash", "-c"]
#RUN pip install "pybind11[global]" && python src/download_model.py
RUN python src/download_model.py

ENV PYTHONPATH "${PYTHONPATH}:${APP_DIR}/src"
# add cpp include path to conda environment so pybind11 header files installed inside conda are found during JIT compilation
#ENV CPLUS_INCLUDE_PATH "${CONDA_PREFIX}/include:${CPLUS_INCLUDE_PATH}"
ENV CPLUS_INCLUDE_PATH "/home/app/.conda/envs/rag/include"
EXPOSE 8000

# The code to run when container is started:
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "rag"]
#ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv", "python3", "src/server.py"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
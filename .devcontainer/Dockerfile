ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}
# here i pick python:3.xx instead of our base on ACR
# becuase the base on ACR is amd64
# then arm machines will have issues
ENV PYTHONUNBUFFERED 1
# to prevent run as root
ARG DOCKER_HOME="/opt/motiong"
ARG DOCKER_CODE="/opt/motiong/code"
ARG DOCKER_GROUP="motiong"
ARG DOCKER_USER="motiong"
ARG DOCKER_UID=5000
ENV ZSH="/opt/motiong/.oh-my-zsh"

RUN mkdir -p ${DOCKER_HOME} ${DOCKER_CODE} && \
    groupadd -g ${DOCKER_UID} ${DOCKER_GROUP} && \
    useradd -r -u ${DOCKER_UID} -g ${DOCKER_GROUP} -d ${DOCKER_HOME} ${DOCKER_USER} && \
    chown -R ${DOCKER_USER}:${DOCKER_GROUP} ${DOCKER_HOME} ${DOCKER_CODE}
COPY .devcontainer/.zshrc ${DOCKER_HOME}/.zshrc
WORKDIR ${DOCKER_HOME}

RUN apt-get update && \
    apt-get install -y curl && \
    apt-get install -y git && \
    apt-get install -y openssh-client && \
    apt-get install -y postgresql-client && \
    pip install --upgrade pip && \
    pip install poetry==1.8.3 && \
    poetry config virtualenvs.create false

# Install zsh, omz, plugins
RUN apt-get install -y zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

WORKDIR ${DOCKER_CODE}
# will add examples to install our own python modules
# maybe not via git+ssh, well we will decide later if we have any
RUN mkdir ${DOCKER_HOME}/.ssh && \
    chown -R ${DOCKER_USER} ${DOCKER_HOME}/.ssh && \
    ssh-keyscan github.com >> ${DOCKER_HOME}/.ssh/known_hosts

ENV PATH "$PATH:/opt/motiong/.local/bin"
ENV PYTHONPATH=.

USER ${DOCKER_USER}
ENTRYPOINT [ "/bin/zsh", "-c"]
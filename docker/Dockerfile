FROM nvidia/cuda:12.2.2-devel-ubuntu22.04


# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

ENV USER_NAME=ismayilz
ENV HOME=/home/${USER_NAME}
ENV CONDA_PREFIX=${HOME}/.conda
ENV CONDA=${CONDA_PREFIX}/condabin/conda
ENV REPO_DIR=${HOME}/project-dycoder
ENV CONDA_ENV=dycoder

WORKDIR /

# Install dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
RUN apt update && apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        cmake \
        git \
        curl \
        vim \
        unzip \
        wget \
        tmux \
        screen \
        ca-certificates \
        apt-utils \
        libjpeg-dev \
        libpng-dev \
        sudo \
        htop \
        apt-transport-https \
        gnupg \
        protobuf-compiler

# Set up user
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    groupadd -g ${GROUP_ID} ${GROUP_NAME} && \
    useradd -r -m -d /home/${USER_NAME} -s /bin/bash -g ${GROUP_ID} -G sudo -u ${USER_ID} ${USER_NAME} && \
    # Change the password to make root > user
    chown ${USER_ID} -R /home/${USER_NAME} && \
    echo -e "${USER_NAME}\n${USER_NAME}" | passwd ${USER_NAME}

RUN echo 'root:patheimathos' | chpasswd
RUN echo -e "\n${USER_NAME} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# Install OpenSSH for MPI to communicate between containers
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd
RUN sed -i 's/#*PermitRootLogin prohibit-password/PermitRootLogin yes/g' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed -i 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' /etc/pam.d/sshd
ENV NOTVISIBLE="in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile
EXPOSE 22

# Prepare the NFS mount folders
RUN mkdir /mnt/scratch

# Git configuration (optional, you can also use a repo saved in the NFS)
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    git config --global user.name ${GITHUB_NAME}
RUN --mount=type=secret,id=my_env source /run/secrets/my_env && \
    git config --global user.email ${GITHUB_EMAIL}
RUN git config --global pull.rebase false

USER ${USER_NAME}

# Switch to home directory
WORKDIR ${HOME}

# Install conda (optional)
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p ${CONDA_PREFIX}
RUN ${CONDA} config --set auto_activate_base false
RUN ${CONDA} init bash
RUN ${CONDA} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN ${CONDA} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN ${CONDA} create -y --name ${CONDA_ENV} python=3.12
RUN ${CONDA} run -n ${CONDA_ENV} pip install ipykernel ipywidgets

# Install dependencies
COPY --chown=${USER_NAME} ./requirements.txt .
RUN --mount=type=cache,id=pip,target=${HOME}/.cache/pip ${CONDA} run -n ${CONDA_ENV} pip install -r requirements.txt

# Transfer scripts
COPY --chown=${USER_NAME} ./docker/entrypoint.sh .
RUN chmod +x ./entrypoint.sh

COPY --chown=${USER_NAME} ./docker/startup.sh .
RUN chmod +x ./startup.sh

ENTRYPOINT ["./entrypoint.sh"]
CMD ["/bin/bash"]
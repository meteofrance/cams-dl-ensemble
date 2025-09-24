# Build from an official pytorch 2.8 image, packaging python 3.11
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Météo France certificates
ARG INJECT_MF_CERT
COPY mf.crt /usr/local/share/ca-certificates/mf.crt
RUN ( test $INJECT_MF_CERT -eq 1 && update-ca-certificates ) || echo "MF certificate not injected"
ARG REQUESTS_CA_BUNDLE
ARG CURL_CA_BUNDLE

# Install necessary packages
RUN apt update && apt install -y sudo git ssh curl openssh-server

# Setup ssh server folder and code server
RUN mkdir -p /run/sshd
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Build variables
ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GUID
ARG HOME_DIR
ARG NODE_EXTRA_CA_CERTS

# Setup user
RUN set -eux && groupadd --gid $USER_GUID $GROUPNAME \
    # https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
    && mkdir -p $HOME_DIR && useradd -l --uid $USER_UID --gid $USER_GUID -s /bin/bash --home-dir $HOME_DIR --create-home $USERNAME \
    && chown $USERNAME:$GROUPNAME $HOME_DIR \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd

# Set work dir
WORKDIR $HOME_DIR

# Install python dependencies
COPY requirements.txt /root/requirements.txt
RUN pip install --upgrade pip && pip install -r /root/requirements.txt

# Install precommit tool
RUN pip install pre-commit

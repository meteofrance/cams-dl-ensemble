# Build from a pytorch 2.8 image.
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Inject Météo France certificates.
ARG INJECT_MF_CERT
COPY mf.crt /usr/local/share/ca-certificates/mf.crt
RUN ( test $INJECT_MF_CERT -eq 1 && update-ca-certificates ) || echo "MF certificate not injected"
ARG REQUESTS_CA_BUNDLE
ARG CURL_CA_BUNDLE

# Download linux dependencies
ENV MY_APT='apt -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

RUN $MY_APT update && $MY_APT install -y curl gcc nano sudo git openssh-server

# Setup code server and ssh for devcontainer capabilities.
RUN mkdir -p /run/sshd
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Setup user.
ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GUID
ARG HOME_DIR
ARG NODE_EXTRA_CA_CERTS
RUN set -eux && groupadd --gid $USER_GUID $GROUPNAME \
    # https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
    && mkdir -p $HOME_DIR && useradd -l --uid $USER_UID --gid $USER_GUID -s /bin/bash --home-dir $HOME_DIR --create-home $USERNAME \
    && chown $USERNAME:$GROUPNAME $HOME_DIR \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd

# Set workdir.
WORKDIR $HOME_DIR

# Install python dependencies.
RUN pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN set -eux && pip install -r requirements.txt

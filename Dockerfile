# Use a Python image with uv pre-installed
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

# Météo France certificates
ARG INJECT_MF_CERT
COPY mf.crt /usr/local/share/ca-certificates/mf.crt
RUN ( test $INJECT_MF_CERT -eq 1 && update-ca-certificates ) || echo "MF certificate not injected"
ARG REQUESTS_CA_BUNDLE
ARG CURL_CA_BUNDLE

# Define apt-get
ENV MY_APT='apt-get -o "Acquire::https::Verify-Peer=false" -o "Acquire::AllowInsecureRepositories=true" -o "Acquire::AllowDowngradeToInsecureRepositories=true" -o "Acquire::https::Verify-Host=false"'

# Install the necessary libraries to use the image as a ssh server host
RUN $MY_APT update && $MY_APT install -y curl gcc g++ nano sudo libgeos-dev libeccodes-dev libeccodes-tools git vim openssh-server wget

# Build time variables
ARG USERNAME
ARG GROUPNAME
ARG USER_UID
ARG USER_GUID
ARG HOME_DIR
ARG NODE_EXTRA_CA_CERTS

# Setup root user
RUN set -eux && groupadd --gid $USER_GUID $GROUPNAME \
    # https://stackoverflow.com/questions/73208471/docker-build-issue-stuck-at-exporting-layers
    && mkdir -p $HOME_DIR && useradd -l --uid $USER_UID --gid $USER_GUID -s /bin/bash --home-dir $HOME_DIR --create-home $USERNAME \
    && chown $USERNAME:$GROUPNAME $HOME_DIR \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && echo "$USERNAME:$USERNAME" | chpasswd

# Setup ssh server
RUN mkdir -p /run/sshd
RUN curl -fsSL https://code-server.dev/install.sh | sh

# Requirements installation
COPY pyproject.toml .
# RUN uv pip install -r pyproject.toml --extra dev --system --break-system-packages
RUN pip install .[dev] --break-system-packages
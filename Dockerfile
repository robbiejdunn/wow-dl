FROM dorowu/ubuntu-desktop-lxde-vnc:bionic-lxqt

RUN sudo dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install --install-recommends -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install --install-recommends -y scrot wget gpg-agent python3.8-dev python3-pip python3.8-venv xdotool

RUN wget -nc https://dl.winehq.org/wine-builds/winehq.key && \
    apt-key add winehq.key && \
    add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ bionic main' && \
    # N0rbert response https://askubuntu.com/questions/1205550/cant-install-wine-from-winehq-org-on-ubuntu-actually-lubuntu-18-04-lts
    wget -q https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/Release.key -O Release.key -O- | sudo apt-key add - && \
    sudo apt-add-repository 'deb https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/ ./' && \
    apt-get update && \
    apt-get install --install-recommends -y winehq-stable

COPY setup.py .
COPY rl_agents /root/rl_agents
RUN python3.8 --version
RUN python3.8 -m venv venv && \
    . venv/bin/activate && \
    python3.8 -m pip install wheel && \
    python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install -e .

COPY docker/Config.wtf /root/client/WTF/

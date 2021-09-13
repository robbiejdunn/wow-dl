FROM dorowu/ubuntu-desktop-lxde-vnc:bionic-lxqt

RUN sudo dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install --install-recommends -y scrot wget gpg-agent python3-dev python3-pip python3-venv

RUN wget -nc https://dl.winehq.org/wine-builds/winehq.key && \
    apt-key add winehq.key && \
    add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ bionic main' && \
    # N0rbert response https://askubuntu.com/questions/1205550/cant-install-wine-from-winehq-org-on-ubuntu-actually-lubuntu-18-04-lts
    wget -q https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/Release.key -O Release.key -O- | sudo apt-key add - && \
    sudo apt-add-repository 'deb https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/ ./' && \
    apt-get update && \
    apt-get install --install-recommends -y winehq-stable

# may need faudio for ubu 18?
# RUN wget https://download.opensuse.org/repositories/Emulators:/Wine:/Debian/xUbuntu_18.04/amd64/libfaudio0_19.07-0~bionic_amd64.deb && \
#     sudo dpkg -i libfaudio0_19.07-0~bionic_amd64.deb

RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install wheel && \
    pip install --upgrade pip && \
    pip install tensorflow && \
    python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))" && \
    # TF agents reqs (from https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial)
    apt-get install --install-recommends -y xvfb ffmpeg && \
    pip install 'imageio==2.4.0' && \
    pip install pyvirtualdisplay && \
    pip install tf-agents matplotlib pyglet

COPY test.txt /root/test.txt

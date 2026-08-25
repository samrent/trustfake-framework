#!/bin/bash

set -euo pipefail

: "${USER:?USER is required}"
: "${USER_ID:?USER_ID is required}"
: "${GROUP_ID:?GROUP_ID is required}"

echo "Starting with UID : $USER_ID, GID: $GROUP_ID, USER: $USER"

echo "Checking uv installation..."
if ! command -v uv &> /dev/null
then
    echo "uv could not be found" # exit there automatically
    exit
else
    echo "uv is installed, version: $(uv --version)"
fi

echo "Setting up user and permissions..."
addgroup -gid "$GROUP_ID" "$USER"
adduser --disabled-password --gecos '' --uid "$USER_ID" --gid "$GROUP_ID" --shell /bin/zsh "$USER"
usermod -aG sudo "$USER"
echo "User $USER with UID $USER_ID and GID $GROUP_ID has been set up."
echo "You can switch to this user with: su $USER"
echo "${USER} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
passwd -d root
echo "Root password has been removed"
chown -R "${USER_ID}:${GROUP_ID}" /home/

echo "Configuring zsh theme for ${USER}..."
if [ -d /root/.oh-my-zsh ]; then
    rm -rf "/home/${USER}/.oh-my-zsh"
    cp -r /root/.oh-my-zsh "/home/${USER}/.oh-my-zsh"
fi

cat > "/home/${USER}/.zshrc" <<EOF
export ZSH="/home/${USER}/.oh-my-zsh"
ZSH_THEME="${ZSH_THEME:-bira}"
plugins=(git)
source \$ZSH/oh-my-zsh.sh
EOF

chown -R "${USER_ID}:${GROUP_ID}" "/home/${USER}/.zshrc" "/home/${USER}/.oh-my-zsh" 2>/dev/null || true
chsh -s $(which zsh) ${USER}

echo "All set! Starting the container..."

# Finally switch to the specified user and execute the command
exec gosu "$USER" "$@"
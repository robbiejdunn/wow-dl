if [ -n "$1" ]; then
    if [ "$1" = "help" ]; then
        echo "Usage: docker_run.sh <directory of game client>"
        exit
    else
        echo "Using game client files from directory: $1"
    fi
else
    echo "Please provide the game client directory as the first argument"
    exit
fi

if [ -n "$2" ]; then
    echo "Using python AI code from directory: $2"
else
    echo "Please provide the python AI code directory as the second argument"
    exit
fi

docker run \
    -p 6080:80 \
    -v /dev/shm:/dev/shm \
    -v "$1/Data:/root/client/Data" \
    -v "$1/WoW.exe:/root/client/WoW.exe" \
    -v "$1/dbghelp.dll:/root/client/dbghelp.dll" \
    -v "$1/DivxDecoder.dll:/root/client/DivxDecoder.dll" \
    -v "$1/fmod.dll:/root/client/fmod.dll" \
    -v "$1/ijl15.dll:/root/client/ijl15.dll" \
    -v "$1/Interface:/root/client/Interface" \
    -v "$1/realmlist.wtf:/root/client/realmlist.wtf" \
    -v "$1/Scan.dll:/root/client/Scan.dll" \
    -v "$1/Scan.dll:/root/client/unicows.dll" \
    -v "$2/docker-out:/root/output" \
    -v "$2/rl_agents:/root/rl_agents" \
    rl-agents-wow

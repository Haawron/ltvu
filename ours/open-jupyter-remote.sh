#!/usr/bin/zsh

set -e

if [[ "$hostname" = "ariel-master" ]]; then
    echo "This script should be run on a compute node, not the head node."
    exit 1
fi

hostname

CONDA_ENV_INHERITED=$CONDA_DEFAULT_ENV
source ~/.zshrc
conda activate $CONDA_ENV_INHERITED
jupyter kernelspec list

# TOKEN=$(openssl rand -hex 32)
TOKEN='89218594e77abd3a0b65943f352cc3fe2496e739ca523bd49b3ef1c26d39c2f0'
TMPDIR=/tmp/$USER
mkdir -p $TMPDIR
PATH_JUPYTER_LOG=$TMPDIR/jupyter_log.txt
JUPYTER_PORT=$(shuf -i 8000-9999 -n 1)  # unique port between 8000 and 9999
jupyter lab \
    --no-browser \
    --NotebookApp.token=$TOKEN \
    --ip=$(hostname -i) \
    --port=$JUPYTER_PORT 2>&1 | tee $PATH_JUPYTER_LOG &

# Wait for Jupyter to start and print the URL
sleep 5
echo -e "\nWaiting for Jupyter to start..."
sleep 5
JUPYTER_URL=$(grep -o "http://[^ ]*" $PATH_JUPYTER_LOG)
# echo $JUPYTER_URL > jupyter_url.txt

ssh -f -N -R 8099:$(hostname -i):$JUPYTER_PORT ariel-master

echo -e "\nSSH tunnel established. Jupyter is running at $JUPYTER_URL\n"
echo -e "Connect to the server through http://localhost:8099/lab?token=${TOKEN}\n"
wait  # for Jupyter to finish

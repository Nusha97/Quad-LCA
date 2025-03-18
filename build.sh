user_id=$(id -u)
user_name=$(whoami)
docker build -t conda_lca --build-arg user_id=$user_id --build-arg user_name=$user_name .

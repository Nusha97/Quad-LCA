FROM continuumio/miniconda3:latest AS miniconda
FROM nvidia/cudagl:11.4.2-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update
# system depends
RUN apt-get install -y --no-install-recommends gcc cmake sudo
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip git python3-dev wget vim curl 

# Add user
ARG user_id
ARG user_name
env USER $user_name
RUN useradd -U --uid ${user_id} -m -s /bin/bash $USER && echo "$USER:$USER" | chpasswd && adduser $USER sudo && echo "$USER ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USER
USER $USER
WORKDIR /home/$USER

# setup conda
COPY lca_sanusha.yml /home/$USER/lca_sanusha.yml
COPY --from=miniconda /opt/conda /opt/conda
RUN sudo chown -R $USER: /opt/conda 
RUN sudo chown -R $USER: /home/$USER
USER $USER
# conda install
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/$USER/.bashrc && \
    echo "conda activate base" >> /home/$USER/.bashrc
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh \
&& conda init bash \
&& conda env create -f lca_sanusha.yml \
&& conda activate lca_sanusha \

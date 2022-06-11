FROM centos:7.9.2009
COPY . .
RUN yum -y install {vim,wget,sox,make,gcc} && \
    sh Anaconda3-2020.07-Linux-x86_64.sh -p anaconda3 -b && \
    mkdir -p anaconda3/envs/huggingface && \
    tar -zxvf hf_pack.tar.gz -C anaconda3/envs/huggingface
ENV PATH anaconda3/bin:$PATH
ENV LD_LIBRARY_PATH anaconda3/bin:$PATH
CMD ["/bin/bash"]
# syntax=docker/dockerfile:1

#This Is A docker file template designed for use with jinja2
#This assumes that the cloud data system is in a like /repo/module/cloud_sys_file
#Its required to specify title, from_image, repo_dir, module, cloud_sys_file, {env_var}, [container_deps], [*args] [ports]

#DOCKER FILE TEMPLATE
#
################################################################################

FROM python:3.9

#Application Specific Constants
WORKDIR /engforge

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

RUN apt-get install -y make automake gcc g++ subversion python3-dev libblas-dev liblapack-dev gfortran


RUN pip install --upgrade pip

COPY ["./engforge", "./engforge"]
COPY ["./docs/build/html","/docs"]

COPY ["./setup.py", "/engforge/"]
COPY ["./README.md", "/engforge/"]
COPY ["./requirements.txt", "/engforge/"]
RUN pip install -e /engforge/

ENTRYPOINT ["/bin/bash"]

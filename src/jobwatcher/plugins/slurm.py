# Copyright 2013-2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance with
# the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
# OR CONDITIONS OF ANY KIND, express or implied. See the License for the specific language governing permissions and
# limitations under the License.
import logging

from common.schedulers.slurm_commands import PENDING_RESOURCES_REASONS, get_pending_jobs_info
from common.utils import check_command_output
from jobwatcher.plugins.utils import get_optimal_nodes

log = logging.getLogger(__name__)


# get nodes requested from pending jobs
def get_required_nodes(instance_properties, max_size):
    log.info("Computing number of required nodes for submitted jobs")
    pending_jobs = get_pending_jobs_info(
        max_slots_filter=instance_properties.get("slots"),
        max_gpus_per_node=instance_properties.get("gpus"),
        max_nodes_filter=max_size,
        filter_by_pending_reasons=PENDING_RESOURCES_REASONS,
    )
    logging.info("Found the following pending jobs:\n%s", pending_jobs)

    slots_requested = []
    nodes_requested = []
    gpu_slots_requested = []
    gpu_nodes_requested = []
    for job in pending_jobs:
        slots_requested.append(job.cpus_total)
        nodes_requested.append(job.nodes)
        gpu_nodes_requested, gpu_slots_requested = _process_gpus_nodes_slots_for_job(
            job, gpu_nodes_requested, gpu_slots_requested
        )
    log.info("list of GPU nodes: {0} \nlist of GPU slots: {1}".format(gpu_nodes_requested, gpu_slots_requested))

    optimal_cpu_nodes = get_optimal_nodes(nodes_requested, slots_requested, instance_properties["slots"])
    optimal_gpu_nodes = get_optimal_nodes(gpu_nodes_requested, gpu_slots_requested, instance_properties["gpus"])
    log.info("optimal nodes for CPUs: {0}".format(optimal_cpu_nodes))
    log.info("optimal nodes for GPUs: {0}".format(optimal_gpu_nodes))

    return max(optimal_cpu_nodes, optimal_gpu_nodes)


def _process_gpus_nodes_slots_for_job(job, gpu_nodes_requested, gpu_slots_requested):
    if job.tres_per_task:
        gpu_nodes_requested.append(job.nodes)
        gpu_slots_requested.append(job.tres_per_task["gpu"] * job.tasks)
        return gpu_nodes_requested, gpu_slots_requested
    if job.tres_per_node:
        gpu_nodes_requested.append(job.nodes)
        gpu_slots_requested.append(job.tres_per_node["gpu"] * job.nodes)
        return gpu_nodes_requested, gpu_slots_requested
    if job.tres_per_job:
        for _ in range(job.tres_per_job["gpu"]):
            gpu_slots_requested.append(1)
            gpu_nodes_requested.append(1)
        return gpu_nodes_requested, gpu_slots_requested

    gpu_nodes_requested.append(job.nodes)
    gpu_slots_requested.append(0)
    return gpu_nodes_requested, gpu_slots_requested


# get nodes reserved by running jobs
def get_busy_nodes():
    command = "/opt/slurm/bin/sinfo -h -o '%D %t'"
    # Sample output:
    # 2 mix
    # 4 alloc
    # 10 idle
    # 1 down*
    output = check_command_output(command)
    logging.info("Found the following compute nodes:\n%s", output.rstrip())
    nodes = 0
    output = output.split("\n")
    for line in output:
        line_arr = line.split()
        if len(line_arr) == 2 and (line_arr[1] in ["mix", "alloc", "down", "down*"]):
            nodes += int(line_arr[0])
    return nodes

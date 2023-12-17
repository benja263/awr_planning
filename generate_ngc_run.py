import os
import argparse
import time
# JOBNAME_PREFIX = 'ml-model.exempt-rl' ml-model.stable-baselines3.exempt-tc-io-gpu
parser = argparse.ArgumentParser(description='Generate bash script for running wandb sweep agent')

parser.add_argument('--instance', default='', help='machine instance type')
parser.add_argument('--dataset_id',type=int, default=0, help='dataset id in ngc')
parser.add_argument('--num_agents',type=int, default=1, help='number of agents to create')
parser.add_argument('--sweep_id', default='', help='wandb sweep id')
parser.add_argument('--git_repo', default='', help='git repo name')
parser.add_argument('--sub_dir', default='', help='inner directory in repository')
parser.add_argument('--label', default='', help='ngc label')
parser.add_argument('--wandb_api_key', default='', help='weights and biases API key')
parser.add_argument('--project_name', default='', help='weights and biases API key')
parser.add_argument('--workspace', default='', help='ngc workspace')
parser.add_argument('--image', default='', help='ngc image')
parser.add_argument('--wandb_username', default='', help='wandb_username')
parser.add_argument('--git_repo_url', default='', help='git_repo_url')
parser.add_argument('--naming', default='', help='naming in ngc')
parser.add_argument('--additional_commands', default='', help='additional_commands', nargs='*')

args = parser.parse_args()
args.additional_commands = args.additional_commands[0]
# print('parsed: {}'.format(args.additional_commands))

if not os.path.exists('scripts'):
    os.makedirs('scripts')

sweep_command = "wandb agent {}/{}/{}".format(args.wandb_username,args.project_name,args.sweep_id)

ngc_template = 'ngc batch run ' \
                   '--team nbu-ai ' \
                   '--instance {} ' \
                   '--ace nv-us-west-2 ' \
                   '--name \'{}\' ' \
                   '--workspace {} ' \
                   '--result /result ' \
                   '--image {} ' \
                   '--label {} ' \
                   '--commandline \'{}\''
# '--datasetid {}:/dataset ' \

bash_prefix = '#!/bin/bash'

def build_command(job_name):
    command = 'export WANDB_API_KEY={} && '.format(args.wandb_api_key)
    # command = command + 'mkdir -p projects/{} && '.format(args.git_repo)
    # command = command + 'git clone {} projects/{} && '.format(args.git_repo_url,args.git_repo)


    if args.sub_dir == 'none':
        # for projects/bla
        command = command + 'cd ' + args.workspace.split(':')[1] + '/projects/{}/ && '.format(args.git_repo)
        # command = command + 'cd /{}/ && '.format(args.git_repo)
    elif args.sub_dir == 'docker_internal':
        command = command + 'cd {}/ && '.format(args.git_repo)
    else:
        command = command + 'cd ' + args.workspace.split(':')[1] + '/projects/{}/{} && '.format(args.git_repo, args.sub_dir)

    if args.additional_commands != 'none':
        command = command + " ".join(args.additional_commands.split('%')) + ' && '

    # command = command + 'git pull && '


    command = command + sweep_command
    # return ngc_template.format(args.giga, args.dataset_id, job_name,args.workspace,args.image, command)
    return ngc_template.format(args.instance, job_name, args.workspace, args.image, args.label, command)


def bash_and_run(bash_commands):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    bash_script = bash_prefix + '\n' + '\n'.join(bash_commands)
    bash_script_name = 'run_{}.sh'.format(timestr)
    file = open('scripts/' + bash_script_name, 'w')
    file.write(bash_script)
    file.flush()
    os.fsync(file.fileno())
    os.system("bash " + 'scripts/' + bash_script_name)


def main():
    bash_commands = list()
    for ii in range(args.num_agents):
        bash_command = build_command('ml-model.{}.exempt-tc-io-gpu {}'.format(args.naming,ii))
        bash_commands.append(bash_command)
    # build bash script and run it
    bash_and_run(bash_commands)
    # print(bash_commands)

if __name__ == '__main__':
    main()

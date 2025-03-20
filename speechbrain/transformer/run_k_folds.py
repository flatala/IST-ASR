import subprocess

# Replace 'ls -la' with any bash command you want to run

commands = []
for i in range(1):
    bash_command = f"python train_k_folds.py hparams/conformer_small_fold_{i+1}.yaml"
    commands.append(bash_command)


for bash_command in commands:
    print('running:', bash_command)
    try:
        result = subprocess.run(bash_command, shell=True, capture_output=True, text=True)
    except Exception as e:
        print(e)
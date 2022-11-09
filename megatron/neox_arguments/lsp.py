import argparse


parser = argparse.ArgumentParser(
            description="GPT-NeoX Configuration", allow_abbrev=False
        )

group = parser.add_argument_group(title="Training Configuration")


group.add_argument(
            "user_script",
            type=str,
            help="User script to launch, followed by any required " "arguments.",
        )

group.add_argument(
            "--conf_dir",
            "-d",
            type=str,
            default=None,
            help="Directory to prefix to all configuration file paths",
        )

group.add_argument(
            "conf_file",
            type=str,
            nargs="+",
            help="Configuration file path. Multiple files can be provided and will be merged.",
        )

args_parsed = parser.parse_args()

print(f'args_parsed: {args_parsed}')
print(f'user_script: {args_parsed.user_script}')

print(f'conf_dir: {args_parsed.conf_dir}')
print(f'conf_file: {args_parsed.conf_file}')


# 按参数顺序传递
# python megatron/neox_arguments/lsp.py generate.py  -d logs a b c

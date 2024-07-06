import argparse as arg


parser = arg.ArgumentParser()

parser.add_argument("--name", type=str)

arg = parser.parse_args()

print(arg.name)


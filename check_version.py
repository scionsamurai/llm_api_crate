import os

# Get the path to the current working directory
current_dir = os.getcwd()

# Construct the path to the Cargo.toml file
cargo_toml_path = os.path.join(current_dir, 'Cargo.toml')

# Read the current Cargo.toml file
with open(cargo_toml_path, 'r') as f:
    current_contents = f.read()

# Check out the previous commit temporarily
os.system('git checkout HEAD~1')

# Read the Cargo.toml file from the previous commit
with open(cargo_toml_path, 'r') as f:
    previous_contents = f.read()

# Check out the current commit again
os.system('git checkout -')

# Extract the version numbers from the files
current_version = current_contents.split('version = "')[1].split('"')[0]
previous_version = previous_contents.split('version = "')[1].split('"')[0]

# Compare the version numbers
version_changed = current_version != previous_version

# Output the result
print(f"::set-output name=version_changed::{version_changed}")

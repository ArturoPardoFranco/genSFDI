echo "Creating environment... "
sleep 2
conda update -n base -c defaults conda -y
conda create --name gensfdi python=3.6 -y
echo "Success! gensfdi environment created."
echo "Run 'conda activate gensfdi' and execute autoinstall.sh"

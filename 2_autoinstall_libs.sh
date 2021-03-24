echo "Installing requirements via pip..."
sleep 2
pip install -r requirements.txt

echo "Installing core-modules..."
sleep 2
cd core-modules
pip install -e .
cd ..

echo "Success!"
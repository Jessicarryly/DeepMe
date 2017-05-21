wget https://physionet.org/challenge/2017/training2017.zip
unzip training2017
wget https://physionet.org/challenge/2017/sample2017.zip
mv sample/validation validation
unzip sample2017
pip install -r requirements.txt

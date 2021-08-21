@REM Install pytorch, cudatoolkit (for Window)
@REM conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
@REM conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
@REM pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

@REM Install pytorch, cudatoolkit (for Linux)
@REM conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
@REM conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
@REM pip3 install torch torchvision torchaudio
@REM pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

@REM Install packages
pip install -r requirements.txt

@REM load pretrained
python load_pretrained.py
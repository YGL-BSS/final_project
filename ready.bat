@REM Install pytorch, cudatoolkit (for Window)
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
@REM conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

@REM Install pytorch, cudatoolkit (for Linux)
@REM conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
@REM conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

@REM Install packages
pip install -r requirements.txt

@REM load pretrained
python load_pretrained.py
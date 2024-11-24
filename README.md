This Repository contains the PyTorch Implementation for the WACV2023 paper:PatchZero: Defending against Adversarial Patch Attacks by Detecting and Zeroing the Patch  (https://arxiv.org/abs/2207.01795)
*Step 1: Prepare VOC datasets and pretrain Faster R-CNN*
    ```cd PatchZero```
    ```python generate_pgdPatch_examples.py```
    It will download PASCAL VOC 2007/2012 train dataset, And train Faster R-CNN using VOC dataset.
    Please modify the save path of checkpoint.
*Step 2: Generate adv examples*
    ```python generate_pgdPatch_examples.py```
    You can modify the parameters(year and image_set) in line 91 to download test and val VOC datasets.
    It will generate adv examples using the checkpoint generated in the previous *Step 1* , and save in ```./adv_examples2007/```.
    If you want to modify the save path, you can read and modify ```class PatchAttack_save``` in ```attack.py```.
*Step 3: The stage 1 of training PSPNet*
    ```git clone https://github.com/Lextal/pspnet-pytorch.git```
    ```python pspnet_stage1.py```
    It will train the PSPNet using generated adv examples in Step 2.
    This code is adapted from ```train.py``` in ```https://github.com/Lextal/pspnet-pytorch.git```
*Step 3.5: Test PSPNet*
    ```python pspnet_stage1_test.py```
    You can check the effectiveness of trained PSPNet, It will output 2 photo in ```output_stage1_test/```.
    Certainly, you can skip this step.
*Step 4: The stage 2 of Fine tuning PSPNet*
    ```python pspnet_stage2.py```
    It will train PSPNet with Faster R-CNN and against BPDA adaptive attack.
    I haven't finished running this code yet, and there are some issues with the implementation of BPDA in ```pspnet_stage2.py (line 51)```.
*Other*
    Of course, without fine-tuning the PSPNET, it can still achieve good results, You can see ```output.png``` and ```outputx.png```.
    Note that, it can only detect patch generate by PGD attack, but cannot detect physically achievable patches!
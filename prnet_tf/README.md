# prnet_tf

Code mostly copied and followed - 

https://github.com/heathentw/prnet-tf2



Download the BFM.mat and BFM_UV.mat and put in /utils/ folder -
[BFM](https://drive.google.com/file/d/1eQzpKKJJnc2MSBo7X-OTgyxibinQnMjL/view?usp=sharing)

Prepare Data - 

Make sure to install mesh_core_cython in the local python environment by running - 

```shell
# directory - face3d/mesh/cython
pip install cython 
python setup.py build_ext -I 
python setup.py install
```

* Point the BFM.mat and BFM_UV.mat in generate_posmap_300WLP.py
* Update the input_path and output_path in generate_posmap_300WLP.py

 


Related Repo - 

Original repo - https://github.com/YadiraF/PRNet 

TF - 

https://github.com/jnulzl/PRNet-Train


torch - 

https://github.com/reshow/PRNet-PyTorch

https://github.com/tomguluson92/PRNet_PyTorch


--------
### Citation

If you use this code, please consider citing:

```
@inProceedings{feng2018prn,
  title     = {Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network},
  author    = {Yao Feng and Fan Wu and Xiaohu Shao and Yanfeng Wang and Xi Zhou},
  booktitle = {ECCV},
  year      = {2018}
}
```
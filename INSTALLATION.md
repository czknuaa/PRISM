# **Dependencies**

Python versions:

* Python 3.8

Operating systems:

* Linux: Ubuntu 18.04+, Centos 7+


Hardware:
NVIDIA RTX GPU
Software:
NVIDIA Driver >= 520

## 0. Install Vulkan
```
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

## 1. Basic Env
First, prepare a conda environment.
```bash
conda create -n PRISM python=3.8
conda activate PRISM
```

```
pip install torch==2.4.1 torchvision sapien==3.0.0b1 scipy==1.10.1 mplib==0.1.1 gymnasium==0.29.1 trimesh==4.4.3 open3d==0.18.0 imageio==2.34.2 pydantic zarr openai huggingface_hub==0.25.0
```

Then, install pytorch3d:
```
cd third_party/pytorch3d_simplified && pip install -e . && cd ../..
```

## 2. Download Assert
```
python ./script/download_asset.py
unzip aloha_urdf.zip && unzip main_models.zip
```

## 3. Modify `mplib` Library Code
### 3.1 Remove `convex=True`
You can use `pip show mplib` to find where the `mplib` installed.
```
# mplib.planner (mplib/planner.py) line 71
# remove `convex=True`

self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=True,
            verbose=False,
        )
=> 
self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            # convex=True,
            verbose=False,
        )
```

### 3.2 Remove `or collide`
```
# mplib.planner (mplib/planner.py) line 848
# remove `or collide`

if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:
                return {"status": "screw plan failed"}
=>
if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:
                return {"status": "screw plan failed"}
```


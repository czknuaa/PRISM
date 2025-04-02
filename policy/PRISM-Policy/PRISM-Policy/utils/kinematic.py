import torch
import pytorch_kinematics as pk
class Kinematic:
    def __init__(self,urdf_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = torch.tensor([[1.0, 0.0, 0.0, 0.13],
                              [0.0, 1.0, 0.0, 0.0],
                              [0.0, 0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]],dtype=torch.float32)
        self.T.to(device = self.device)
        with open(urdf_path, 'rb') as f:
            urdf_bytes = f.read()
            self.left_chain = pk.build_serial_chain_from_urdf(urdf_bytes,end_link_name="fl_link6",root_link_name="fl_base_link").to(device=self.device)
            self.right_chain = pk.build_serial_chain_from_urdf(urdf_bytes,end_link_name="fr_link6",root_link_name="fr_base_link").to(device=self.device)
     
    def get_pos_distances(self,end_pred_l_pose,end_pred_r_pose,end_traj_l_pose,end_traj_r_pose):
        trans_pred_l = end_pred_l_pose[:, :3, 3]
        trans_pred_r = end_pred_r_pose[:, :3, 3]
        trans_traj_l = end_traj_l_pose[:, :3, 3]
        trans_traj_r = end_traj_r_pose[:, :3, 3]
        l_pos_distances = torch.sum((trans_pred_l - trans_traj_l) ** 2, dim=1)
        r_pos_distances = torch.sum((trans_pred_r - trans_traj_r) ** 2, dim=1)
        return l_pos_distances,r_pos_distances
    def get_orientation_error(self,end_pred_l_pose,end_pred_r_pose,end_traj_l_pose,end_traj_r_pose):
        rot_pred_l = end_pred_l_pose[:, :3, :3]
        rot_pred_r = end_pred_r_pose[:, :3, :3]
        rot_traj_l = end_traj_l_pose[:, :3, :3]
        rot_traj_r = end_traj_r_pose[:, :3, :3]
        rot_diff_l = torch.sum((rot_pred_l-rot_traj_l)**2,dim=(-2, -1))
        rot_diff_r = torch.sum((rot_pred_r-rot_traj_r)**2,dim=(-2, -1))
      
        return rot_diff_l, rot_diff_r                    

    def cal_loss_in_cartesian(self,q_pred,q_traj):
        
        Ts = self.T.unsqueeze(0).expand(q_pred.shape[0], -1, -1).to(device=self.device)
        end_pred_l_pose = torch.bmm(self.left_chain.forward_kinematics(q_pred[:,0:6]).get_matrix(),Ts)
        end_pred_r_pose = torch.bmm(self.right_chain.forward_kinematics(q_pred[:,7:13]).get_matrix(),Ts)
        end_traj_l_pose = torch.bmm(self.left_chain.forward_kinematics(q_traj[:,0:6]).get_matrix(),Ts)
        end_traj_r_pose = torch.bmm(self.right_chain.forward_kinematics(q_traj[:,7:13]).get_matrix(),Ts)
     
        l_pos_distances, r_pos_distances = self.get_pos_distances(end_pred_l_pose,end_pred_r_pose,end_traj_l_pose,end_traj_r_pose)
        angle_error2_l, angle_error2_r = self.get_orientation_error(end_pred_l_pose,end_pred_r_pose,end_traj_l_pose,end_traj_r_pose)
        grip_error2_l = (q_pred[:,6]-q_traj[:,6])**2
        grip_error2_r = (q_pred[:,13]-q_traj[:,13])**2
        
        loss_in_cartesian = torch.stack((l_pos_distances,angle_error2_l,grip_error2_l,r_pos_distances,angle_error2_r,grip_error2_r),dim=1)
        return loss_in_cartesian
    
    def cal_normal_loss_in_catresian(self,q_pred,q_traj,pos_weight = 10.0,rot_weigth = 1.0,grip_weight = 10.0):
        loss_in_cartesian = self.cal_loss_in_cartesian(q_pred,q_traj)
        weight = torch.tensor([pos_weight, rot_weigth, grip_weight, pos_weight, rot_weigth, grip_weight],dtype=torch.float32).to(device=self.device)  # 每一列的除数
        weight = weight.unsqueeze(0).repeat(loss_in_cartesian.shape[0], 1)  # 扩展为 B x 6 的形状
        return loss_in_cartesian*weight
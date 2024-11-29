import open3d as o3d
import copy
import numpy as np
import math


def save_view_point(pcd, filename):  #计算多视角变换中的外参矩阵
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd', width=800, height=600)
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):   #批量实现点云多视角变换

   param = o3d.io.read_pinhole_camera_parameters(filename)
   pcd_temp = copy.deepcopy(pcd)
   params = param.extrinsic
   pcd_temp.transform(params)
   o3d.io.write_point_cloud("D:/pythonproject/Pointnext/temp/temp.ply", pcd_temp,write_ascii=True)




if __name__ == "__main__":
    #path = "D:/pythonproject/Pointnext/temp/temptemp.ply"
    #pcd = o3d.io.read_point_cloud(path)  # 传入自己当前的pcd文件
    #save_view_point(pcd, "D:/pythonproject/Pointnext/temp/viewpoint.json")  # 保存好得json文件位置
    # load_view_point(pcd, "D:/pythonproject/Pointnext/temp/viewpoint2.json")  # 加载修改时较后的pcd文件

    # newpcd = o3d.geometry.PointCloud()
    # newpcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    # load_view_point(pcd,"D:/pythonproject/Pointnext/temp/viewpoint2.json")
    # pcd.estimate_normals(       #计算法向量
    #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30)
    # )
    # print(np.asarray(pcd.normals).shape)
    #o3d.io.write_point_cloud("D:/pythonproject/Pointnext/temp/temptemp.ply", newpcd, write_ascii=True)


    #计算外参
    temppath = "D:/pythonproject/Pointnext/temp/Run1_WT1_111.txt"
    matraix = np.loadtxt(temppath)
    matraix = matraix[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(matraix)
    save_view_point(pcd, "D:/pythonproject/Pointnext/temp/viewpoint.json")  # 保存好得json文件位置

    #转换多视角
    # temppath = "D:/pythonproject/Pointnext/temp/Run1_WT1_9.txt"
    # matraix = np.loadtxt(temppath)
    # matraix = matraix[:, :3]
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(matraix)
    # load_view_point(pcd, "D:/pythonproject/Pointnext/temp/viewpoint2.json")


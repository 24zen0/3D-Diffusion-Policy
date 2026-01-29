workspace=$(pwd)

# 启动第一个 follow1 文件
# gnome-terminal -- bash -c "cd ${workspace}/follow1 && source devel/setup.bash && roslaunch arm_control arx5v.launch"
gnome-terminal -- bash -c "cd /home/dc/Desktop/ARX_Remote_Control/follow1 && source devel/setup.bash && roslaunch arm_control arx5v.launch;exec bash"


sleep 0.5
# gnome-terminal -- bash -c "cd ${workspace}/follow1 && source devel/setup.bash && roslaunch arm_control camera.launch"
gnome-terminal -- bash -c "cd /home/dc/Desktop/ARX_Remote_Control/follow1 && source devel/setup.bash && roslaunch arm_control camera.launch;exec bash"

#!/usr/bin/env python3
import sys
sys.path.append("/home/dc/anaconda3/envs/dc/lib/python3.8/site-packages")
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np
import cv2

# 配置相机深度流
def configure_camera(serial_number):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    # 启用深度流（640x480，Z16格式，30fps）
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    try:
        pipeline.start(config)
        rospy.loginfo("Depth camera with serial number %s started successfully", serial_number)
        return pipeline
    except Exception as e:
        rospy.logerr("Failed to start depth camera: %s", str(e))
        return None

def main():
    try:
        rospy.loginfo("Starting depth camera publisher")
        serial_number_dict = {'my_camera': '317622070255'}
        rospy.loginfo(f"Attempting to connect to depth camera with serial: {serial_number_dict['my_camera']}")
    
    # 初始化深度相机
        Depth_pipeline = configure_camera(serial_number_dict['my_camera'])
        if Depth_pipeline is None:
            rospy.logerr("Exiting due to camera initialization failure")
            return
        rospy.loginfo("Depth camera pipeline configured successfully")

    # 初始化ROS节点
        rospy.init_node('depth_camera_publisher')
        rospy.loginfo("ROS node initialized")
        bridge = CvBridge()
        
        # 创建深度图像发布者
        depth_pub = rospy.Publisher('/depth_camera', Image, queue_size=10)
        rospy.loginfo("Depth image publisher created")
        window_name = "Depth Camera Feed"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 640, 480)
        rospy.loginfo("OpenCV window for depth feed created")   

        rate = rospy.Rate(10)  # 发布频率10Hz
        frame_count = 0
        while not rospy.is_shutdown():
            try:
                frames = Depth_pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                
                if not depth_frame:
                    continue
                    
                # 将深度数据转换为numpy数组
                depth_data = np.asanyarray(depth_frame.get_data())
                # 归一化处理用于显示
                depth_min = depth_data.min()
                depth_max = depth_data.max()
                if depth_max > depth_min:
                    depth_image = np.uint8((depth_data - depth_min) / (depth_max - depth_min) * 255)
                else:
                    depth_image = np.zeros_like(depth_data, dtype=np.uint8)

                cv2.imshow(window_name, depth_image)
                # 检查按键事件（必须调用waitKey以更新窗口）
                key = cv2.waitKey(1)
                if key == ord('q') or key == 27:  # 按q或ESC退出
                    rospy.loginfo("User requested exit")
                    break
            
            # 创建ROS图像消息（16UC1编码对应深度数据）
                depth_msg = bridge.cv2_to_imgmsg(depth_data, encoding="16UC1")
                depth_msg.header.stamp = rospy.Time.now()
                depth_msg.header.frame_id = f"depth_camera_frame_{frame_count}"
                depth_pub.publish(depth_msg)
                rospy.loginfo(f"Published depth frame {frame_count}")
                frame_count += 1
            except Exception as e:
                rospy.logerr(f"Error capturing or publishing depth frame: {e}")
                break
            
            rate.sleep()
    except Exception as e:
        rospy.logerr(f"Unexpected error: {e}")
    finally:
        cv2.destroyAllWindows()   
    Depth_pipeline.stop()

if __name__ == '__main__':
    main()